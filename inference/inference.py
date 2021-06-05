import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2


class Inference:
    
    def __init__(self,weight_path) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.resnet34(pretrained=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=7)
        self.checkpoint = torch.load(weight_path,map_location = torch.device(self.device))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.labels = ['Rs 10','Rs 20','Rs 50','Rs 100','Rs 200','Rs 500','Rs 2000']
        self.model = self.model.to(self.device)
        
    def run_image(self,path,show=True):
        
        img = Image.open(path)
        
        if img.mode != 'RGB':
                img = img.convert('RGB')
        
        img = transforms.Resize(size=(224, 224))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(img)
            sft = torch.nn.Softmax(dim=1)
            s_pred = sft(prediction)
            indx = torch.argmax(prediction)
            prob = s_pred[0][indx]
        
        if prob > 0.75:
            label = self.labels[indx] + f', Prob : {round(prob.item()*100,2)}'
        else:
            label = 'No Currency'
            
        self.result = f'Predicted Indian Currency : {label}'
        
        if show == True: 
            im = plt.imread(path)
            plt.imshow(im)
            plt.axis('off')
            plt.title(self.result)
        
    def return_result(self):
        
        return self.result
      
    
    def predict(self,image):
        
        img = Image.fromarray(image)
        img = transforms.Resize(size=(224, 224))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(img)
            sft = torch.nn.Softmax(dim=1)
            s_pred = sft(prediction)
            indx = torch.argmax(prediction)
            prob = s_pred[0][indx]
            
        if prob > 0.75:
            label = self.labels[indx] + f', Prob : {round(prob.item()*100,2)}'
        else:
            label = 'No Currency'
            
        return label
    
    def run_video(self,path=0):
        
        if path == 0:
            # webcame
            vid = cv2.VideoCapture(0)
            vid_cod = cv2.VideoWriter_fourcc(*'XVID')
            _outmet = True
            while(True):
                _, frame = vid.read()
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                new_h = int(height / 1.1)
                new_w = int(width / 1.1)
                
                if _outmet == True:
                    output = cv2.VideoWriter("cam_video.mp4", vid_cod,20, (new_w, new_h))
                    _outmet = False
                    
                frame = cv2.resize(frame, (new_w, new_h))
                
                x_sp,y_sp = int(frame.shape[0]/2.5),int(frame.shape[1]/2.5)
                x_ep,y_ep = int(frame.shape[0]/3 + 330),int(frame.shape[1]/3 + 170)
                
                sp = (x_sp,y_sp)
                ep = (x_ep,y_ep)
                cx = x_sp
                cw = x_ep - x_sp
                cy = y_sp
                ch = y_ep - y_sp
                
                _pframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop_roi = _pframe[cy:cy+ch,cx:cx+cw,:]
                
                label = self.predict(crop_roi)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.imshow('roi', crop_roi)
                _frame = cv2.rectangle(frame,sp,ep,(50,150,100),2)
                
                if label != 'No Currency':
                    _frame = cv2.putText(_frame,label,sp, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    output.write(_frame)
                output.write(_frame)

                cv2.imshow('frame', _frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # After the loop release the cap object
            vid.release()
            output.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            
        
        
        
        
        
        
        
        
            
        
    