# Based on
# https://www.le arnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

# Import torch and torchvision modules
import torch
from torchvision import models, transforms
import torch.onnx
from PIL import Image  # Import Pillow
from typing import Tuple, List
import onnxruntime
import numpy as np


class AlexNetInference:
    def __init__(self) -> None:
        # Load the pre-trained model from a file
        self.alexnet = torch.load("alexnet-pretrained.pt")
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # # Optional: export model to onnx
        # x = torch.randn(1, 3, 224, 224, requires_grad=True)
        # torch.onnx.export(self.alexnet, x, "alexnet-pretrained.onnx")
        self.onnxalexnet = onnxruntime.InferenceSession("alexnet-pretrained.onnx")

        with open("imagenet_classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def evaluate(self, image_path: str, results_num: int) -> List[Tuple]:
        # Load the test image.
        image = Image.open(image_path)

        image_t = self.transform(image)
        batch_t = torch.unsqueeze(image_t, 0)
        print(batch_t)
        self.alexnet.eval()
        out = self.alexnet(batch_t)

        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        return [
            (self.classes[idx], percentage[idx].item()) for idx in indices[0][:results_num]
        ]


    def evaluate_onnx(self, image_path: str, results_num: int) -> List[Tuple]:
        # Load the test image.
        image = Image.open(image_path)

        image_t = self.transform(image)
        batch_t = torch.unsqueeze(image_t, 0)

        ort_inputs = {self.onnxalexnet.get_inputs()[0].name: batch_t.cpu().numpy().astype('float32')}
        ort_outs = self.onnxalexnet.run(None, ort_inputs)

        out = torch.from_numpy(ort_outs[0])

        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        return [
            (self.classes[idx], percentage[idx].item()) for idx in indices[0][:results_num]
        ]


    def evaluate_formatted(self, image_path, results_num=5) -> str:
        results = self.evaluate(image_path, results_num)
        output = "\n".join(["\t" + str(r) for r in results])
        return f"Return top {results_num} inference results:\n" + output

    def evaluate_formatted_onnx(self, image_path, results_num=5) -> str:
        results = self.evaluate_onnx(image_path, results_num)
        output = "\n".join(["\t" + str(r) for r in results])
        return f"Return top {results_num} inference results:\n" + output

if __name__ == "__main__":
    # output_filename = "alexnet-pretrained.pt"
    # alexnet = models.alexnet(pretrained=True)
    # torch.save(alexnet, output_filename)

    alexnet = AlexNetInference()
    # output = alexnet.evaluate_formatted("strawberries.jpg")
    # print(output)
    # output = alexnet.evaluate_formatted_onnx("strawberries.jpg")
    # print(output)

    output = alexnet.evaluate_formatted("automotive.jpg")
    print(output)
    output = alexnet.evaluate_formatted_onnx("automotive.jpg")
    print(output)
