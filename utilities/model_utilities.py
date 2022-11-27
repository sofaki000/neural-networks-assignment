import torch
from tqdm import tqdm

def save_model_on_path(model, path):
    torch.save(model.state_dict(), path)

def validate_one_epoch(model, testloader,loss_fn):
    model.eval()
    eval_losses = []
    eval_accu = []
    running_loss = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            correct = 0
            total = 0
            images, labels = data
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1) #same probably: torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            test_loss = running_loss / len(testloader)
            accu = 100. * correct / total

    print('Validation Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))
    return test_loss, accu

def test(model, testloader,loss_fn):
    model.eval()
    eval_losses = []
    eval_accu = []
    running_loss = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            correct = 0
            total = 0

            images, labels = data
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1) #same probably: torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            test_loss = running_loss / len(testloader)
            accu = 100. * correct / total
            eval_accu.append(accu)
            eval_losses.append(test_loss)

    # eval_losses.append(test_loss)
    # eval_accu.append(accu)
            print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))
    return eval_losses, eval_accu







    # def test(testloader):
    #     correct = 0
    #     total = 0
    #     # since we're not training, we don't need to calculate the gradients for our outputs
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data
    #             # calculate outputs by running images through the network
    #             outputs = model(images)
    #             # the class with the highest energy is what we choose as prediction
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             accuracy_per_batch = 100. * correct / total
    #     print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
