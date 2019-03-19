import torch
import random
import torch.nn.functional as F

def softmax(x):
    e = torch.exp(x - torch.max(x))
    return e / torch.sum(e)
def my_sigmoid(x): # Производная в нуле равна 1
    return 40 / (1 + torch.exp(-0.1 * x))

class Network:
    def __init__(self, nominals, layersCount, alpha):
        self.nominals = torch.tensor(nominals, dtype=torch.float64, requires_grad=True)
        self.layers = [torch.tensor([10] * len(nominals), dtype=torch.float64, requires_grad=True) for i in range(layersCount)]
        self.alpha = alpha
    def clear_gradient(self):
        self.nominals.detach()
        self.nominals.requires_grad_(True)
        for i in range(len(self.layers)):
            self.layers[i].detach()
            self.layers[i].requires_grad_(True)
    def get_p(self, x):
        return softmax(self.layers[0] + torch.sigmoid(x * self.layers[1]) * self.layers[2])
    def update_layers(self):
        with torch.no_grad():
            for i in range(len(self.layers)):
                self.layers[i] = self.layers[i] - self.alpha * self.layers[i].grad
    def teach(self, x):
        self.clear_gradient()
        x1 = x
        p = self.get_p(x)
        while True:
            p = self.get_p(x1)
            const_alpha = 1
            delta = torch.sum(self.nominals * (p ** const_alpha)) / torch.sum(p ** const_alpha)
            if x1 - delta <= 0:
                break
            x1 = x1 - delta
        loss = 1 / len(self.nominals) * torch.sum((x1 - self.nominals) ** 2 * p)
        loss.backward()
        self.update_layers()
        return loss.item()

if __name__ == "__main__":
    random.seed(1)
    result = []
    nominals = [1, 2, 5, 10]
    network = Network(nominals, 3, 0.01)
    for i in range(1000):
        x = random.randint(0, 100)
        if (x == 50):
            result.append((i, network.teach(x)))
        else:
            network.teach(x)
   #     if i % 100 == 0:
           # print(network.layers)
    print(result)
    print(network.get_p(5000))
    print(nominals)
    test = 1
    while (test != -100):
        test = int(input())
        while test > 0:
            print(network.get_p(test))
            test -= nominals[int(input())]
            print(test)
