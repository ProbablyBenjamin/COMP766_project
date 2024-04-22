
import torchvision
import torch
import matplotlib.pyplot as plt
from preprocess import get_train_test_datasets
from model import get_model, save_model, load_model_weights

SAVE_PATH = '/Users/benlo/repo/COMP766_project/saved_models'
train_loader, validation_loader, test_loader = get_train_test_datasets(0.8,0.1,32)

# Run this to test your data loader
#images, labels = next(iter(train_loader))
# helper.imshow(images[0], normalize=False)
#print(labels)
#plt.imshow(images[1].permute(1,2,0), vmin=0, vmax=1)
#plt.show()


#get model, optimizer and loss criterion
model, criterion, optimizer = get_model(lr=0.01)

#train model

#epochs = 1
#model.train()

#for epoch in range(epochs):
    #running_loss = 0.0
    #if epoch == 30:
        #torch.optim.param_groups[0]['lr'] = 0.1
    #for images, labels in train_loader:
        #optimizer.zero_grad()
        #outputs = model(images)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()

        #print(loss.item())
        #running_loss += loss.item()

    #save_model(epoch + 1, model, path=f'{SAVE_PATH}/model_epoch_{epoch + 1}')
    #print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in validation_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')