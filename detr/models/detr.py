# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import random
from math import sqrt
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from detr.util import box_ops
from detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer





class MultiTensorScaling(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, scale, *param_list):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(*param_list, scale.detach())
        return torch._foreach_mul(param_list, float(scale))

    @staticmethod
    def backward(ctx, *grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        *param_list, scale = ctx.saved_tensors
        dscale_dL = sum(grad.sum() for grad in torch._foreach_mul(param_list, grad_output))
        dparam_dL = torch._foreach_mul(grad_output, float(scale))
        return (dscale_dL,) + dparam_dL


multitensor_scaling = MultiTensorScaling.apply

x = [torch.randn([1], dtype=torch.float64).view([]), torch.randn(3, 3, dtype=torch.float64),
     torch.randn(4, 5, dtype=torch.float64), torch.randn(3, 3, dtype=torch.float64)]
for p in x:
    p.requires_grad = True
#torch.autograd.gradcheck(lambda scale, a, b, c: sum(p.sum() for p in multitensor_scaling(scale, a, b, c)), x)
torch.autograd.gradcheck(multitensor_scaling, x)
#exit()

def assign_scaled_parameter(module, scaled_param_dict, prefix=""):
    module._parameters_backup = module._parameters
    module._buffers_backup = module._buffers
    module._parameters = {}
    module._buffers = {}
    if prefix:
        prefix = prefix + "."
    # Set the weight to the scaled value if present, otherwise use the
    # original value.
    for name, parameter in module._parameters_backup.items():
        key = f"{prefix}{name}"
        module.__setattr__(name, scaled_param_dict.get(key, parameter))
    for name, buffer in module._buffers_backup.items():
        key = f"{prefix}{name}"
        module.__setattr__(name, scaled_param_dict.get(key, buffer))
    for name, submodule in module._modules.items():
        assign_scaled_parameter(submodule, scaled_param_dict, prefix=f"{prefix}{name}")



# def scale_parameter_member(module, scale, mean=None):
#     module._parameters_backup = module._parameters
#     module._buffers_backup = module._buffers
#     module._parameters = {}
#     module._buffers = {}
#     for name, parameter in module._parameters_backup.items():
#         if parameter is None or "global_scale" in name:
#             value = parameter
#         elif mean is None:
#             value = parameter * scale
#         else:
#             value = (parameter - mean) * scale
#         module.__setattr__(name, value)
#     for name, buffer in module._buffers_backup.items():
#         if buffer is None or "global_scale" in name:
#             value = buffer
#         elif mean is None:
#             value = buffer * scale
#         else:
#             value = (buffer - mean) * scale
#         module.__setattr__(name, value)
#     for name, submodule in module._modules.items():
#         if "backbone" not in name:
#             scale_parameter_member(submodule, scale, mean)

def restore_params(module):
    module._parameters = module._parameters_backup
    module._buffers = module._buffers_backup
    for name, submodule in module._modules.items():
        restore_params(submodule)
    #module._parameters_backup = None
    #module._buffers_backup = None


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, high_def=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.high_def = high_def
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if self.high_def:
            self.projections = torch.nn.ModuleList([nn.Conv2d(num_channels, hidden_dim, kernel_size=1) for num_channels in backbone.num_channels])
        else:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        #self.bootstrap_steps = [10, 20]
        self.bootstrap_steps = [10]
        self.epoch = 0
        self.stride = None
        self.register_parameter("global_scale", nn.Parameter(torch.Tensor([0.022])))
        # Scale only once.
        self.scaled = False

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # Global weight norm.
        #params = [p for n, p in self.named_parameters() if ("backbone" not in n and "global_scale" not in n) and p.requires_grad]
        #params = [p for p in self.parameters()]
        #contiguous = torch.cat([p.view(-1) for p in params])
        #mean = contiguous.mean()
        #std = contiguous.std()
        # Initial std: 0.0392
        #print("\n\nstd", std, "\n\n")
        #scale_parameter_member(self, 0.022 / std, mean)
        #scale_parameter_member(self, 0.022 / std)
        #scale_parameter_member(self, self.global_scale / std)
        if not self.scaled:
            self.scaled = True
            ignore = ["backbone", "global_scale", "bbox_embed.layers.2", "query_embed.weight", "class_embed."]
            named_params = [(n, p) for n, p in self.named_parameters() if all(i not in n for i in ignore) and p.requires_grad]
            names, params = zip(*named_params)
            contiguous = torch.cat([p.view(-1) for p in params])
            std = contiguous.std()
            with torch.no_grad():
                for param in params:
                    param.data.mul_(0.022 / std)
        # scaled_params = multitensor_scaling((0.022 / std), *params)
        
        # # param_dict = dict(named_params)
        # # scaled_params = []
        # # for (n, p) in named_params:
        # #     # Replace bias with weight for fan in computation.
        # #     fanin_name = n
        # #     if n.endswith("bias"):
        # #         scaled_params.append(p)
        # #         fanin_name = n[:-4] + "weight"
        # #     fanin_shape = param_dict[fanin_name].shape
        # #     fanin = fanin_shape.numel() / fanin_shape[0]
        # #     scale = sqrt(1/fanin) / std
        # #     print(fanin, float(std), sqrt(1/fanin), float(scale), float((p*scale).std()))
        # #     scaled_params.append(p * scale)
                
        
        # #scaled_params = multitensor_scaling((self.global_scale.squeeze() / std), *params)
        # scaled_param_dict = dict(zip(names, scaled_params))
        
        # # a_names, attention_params = zip(*[(n, p) for n, p in named_params if "in_proj" in n])
        # # rest_names, rest_params = zip(*[(n, p) for n, p in named_params if "in_proj" not in n])
        
        # # #a_scaled_params = multitensor_scaling(sqrt(0.022) / std.sqrt(), *attention_params)
        # # a_scaled_params = multitensor_scaling((0.022 / std).sqrt(), *attention_params)
        # # rest_scaled_params = multitensor_scaling(0.022 / std, *rest_params)
        
        # # scaled_param_dict = dict(zip(a_names, a_scaled_params))
        # # scaled_param_dict.update(dict(zip(rest_names, rest_scaled_params)))
        # assign_scaled_parameter(self, scaled_param_dict)

       

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        if self.high_def:
            # Use the middle feature map as reference
            reference = 1
            pos = pos[reference]
            src, mask = features[reference].decompose()
            src = self.projections[reference](src)
            for i,(projection, fmap) in enumerate(zip(self.projections, features)):
                if i == reference:
                    continue
                fmap = projection(fmap.tensors)
                src += torch.nn.functional.interpolate(fmap, src.shape[2:4],
                                                       mode="bilinear", align_corners=True)
        else:
            pos = pos[-1]
            src, mask = features[-1].decompose()
            assert mask is not None
            src = self.input_proj(src)

        if self.bootstrap_steps and self.training:
            stride = sum([self.epoch < e for e in self.bootstrap_steps]) + 1
            self.stride = stride
            if stride > 1:
                x_i, y_i = random.choices(range(stride), k=2)
                src = src[:, :, y_i::stride, x_i::stride]
                mask = mask[:, y_i::stride, x_i::stride]
                pos = pos[:, :, y_i::stride, x_i::stride]
        
        hs = self.transformer(src, mask, self.query_embed.weight, pos)[0]
        #hs = hs[:, :, :self.num_queries]
        outputs_class = self.class_embed(hs)
        #outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_coord = self.bbox_embed(hs)
        outputs_coord[..., 2:] = torch.exp(outputs_coord[..., 2:])
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        #restore_params(self)
        return out

    @torch.jit.unused 
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Set target class to background if iou < 0.2
        #src_boxes = outputs['pred_boxes'][idx]
        #target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #iou = box_ops.pairwise_box_iou(src_boxes, target_boxes)[0]
        #target_classes_o[iou < 0.2] = self.num_classes

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
    
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        high_def=args.high_def
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
