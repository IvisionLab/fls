import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from rboxnet import model, base, utils
from rboxnet.model import log


class Inference(base.Base):
  def __init__(self, config):
    super().__init__(self.build(config), config)

  def build(self, config):
    # Check image size
    base.check_image_size(config)

    # input image layer
    input_image = KL.Input(
        shape=config.IMAGE_SHAPE.tolist(), name="input_image")

    # Feature extractor layer
    [P2, P3, P4, P5, P6] = base.feature_extractor_layers(input_image, config)
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    rbox_feature_maps = [P2, P3, P4, P5]

    # Generate Anchors
    self.anchors = utils.generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
        config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE)

    # RPN layer
    rpn_class_logits, rpn_class, rpn_bbox, rpn_rois = base.rpn_layers(
        rpn_feature_maps, self.anchors, config)

    inputs = [input_image]
    outputs = [rpn_rois, rpn_class, rpn_bbox]

    return KM.Model(inputs, outputs, name='rboxnet')

  def detect(self, images, verbose=0):
    assert len(images) == self.config.BATCH_SIZE, \
        "len(images) must be equal to BATCH_SIZE"

    if verbose:
      log("Processing {} images".format(len(images)))
      for image in images:
        log("image", image)

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = self.mold_inputs(images)

    if verbose:
      log("molded_images", molded_images)
      log("image_metas", image_metas)

    rpn_rois, rpn_class, rpn_bbox = self.keras_model.predict([molded_images],
                                                             verbose=0)

    return [rpn_rois, rpn_class, rpn_bbox]