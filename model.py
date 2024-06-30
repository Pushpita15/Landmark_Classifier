import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.featureExtractor=nn.Sequential(
            
            #first conv layer + maxpool + dropout
            nn.Conv2d(3,16,3,padding=1),#224*224*16
            nn.MaxPool2d(2,2),#112*112*16
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            
            #second conv layer + maxpool + dropout
            nn.Conv2d(16,32,3,padding=1),#112*112*32
            nn.MaxPool2d(2,2),#56*56*32
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            
            #third conv layer + maxpool + dropout
            nn.Conv2d(32,64,3,padding=1),#56*56*64
            nn.MaxPool2d(2,2),#28*28*64
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            
            nn.Flatten(),#1*28*28*64
        )
        self.Classifier = nn.Sequential(
            
            nn.Linear(50176,25088),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(25088,num_classes)
        )
        
       


            
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.featureExtractor(x)
        x = self.Classifier(x)
#         print(x.shape)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
