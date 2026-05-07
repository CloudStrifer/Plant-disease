import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class SeverityHead(nn.Module):
    def __init__(self, in_features: int, num_grades: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_grades)

    def forward(self, x):
        return self.fc(x)


class ConceptHead(nn.Module):
    def __init__(self, in_features: int, num_concepts: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_concepts)

    def forward(self, x):
        return self.fc(x)
