from omegaconf import OmegaConf

from ..config import TransformConfig
from .base import BaseTransform


# Use below Compose when using transforms has multi input.
class MultiCompose:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, img, data):
        for t in self.transforms:
            img, data = t(img, data)
        return img, data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class BatchedTransformCompose:
    def __init__(self, cfg: list[TransformConfig]):
        """このクラスは、設定ファイルから変換操作のリストを作成し、それらを順番に適用します。
        バッチ処理に対応した変換の適用と、assign_label機能をサポートしています。

        Attributes:
            transforms (list): 適用する変換操作のリスト。
            assign_labels (list): 各変換操作に対応するassign_label設定のリスト。各要素は
                                  データのキーのリストまたはNoneです。

        Note:
            assign_labelについて:
            - assign_labelは、変換操作の結果を特定のデータキーに割り当てる機能です。
            - 設定ファイルでassign_labelにキーのリストを指定することで有効になります。
            - 変換操作がassign_label属性を持つ場合、その結果が指定されたデータキーに適用されます。
            - これにより、画像変換と同時に指定されたデータも一貫して変換することができます。

        Example:
            cfg = [
                TransformConfig(name='RandomRotation', args={'degrees': 30, 'assign_label': ['mask', 'keypoints']}),
                TransformConfig(name='RandomHorizontalFlip', args={'p': 0.5, 'assign_label': ['mask']})
            ]
            compose = BatchedTransformCompose(cfg)
            transformed_data = compose(data)

        Note:
            - assign_label機能を使用する変換操作は、その機能をサポートするように
              実装されている必要があります。
            - すべての変換操作がassign_label機能をサポートしているわけではありません。
            - assign_labelがNoneまたは空リストの場合、その変換操作はデータに適用されません。
        """
        from .build import BATCHED_TRANSFORM_REGISTRY

        self.transforms = []
        self.assign_labels = []

        for c in cfg:
            assign_label = c.args.get("assign_label", False)
            transform = self._create_transform(c, BATCHED_TRANSFORM_REGISTRY)
            self.transforms.append(transform)

            if hasattr(transform, "assign_label") and assign_label:
                self.assign_labels.append(transform.assign_label)
            else:
                self.assign_labels.append(False)

    def _create_transform(self, config: TransformConfig, registry):
        if config.args is None:
            return registry.get(config.name)()
        else:
            args = OmegaConf.to_object(config.args)
            if args.get("assign_label", False):
                args.pop("assign_label")
            return registry.get(config.name)(**args)

    def __call__(self, data):
        for i, t in enumerate(self.transforms):
            data["image"] = t(data["image"])
            if self.assign_labels[i]:
                for k in self.assign_labels:
                    data[k] = t(data[k], t._params)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
