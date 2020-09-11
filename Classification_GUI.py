import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2

#load the trained model to classify the images
from keras.models import load_model
import tensorflow_hub as hub
model = load_model('model_20_epochs.h5', custom_objects={'KerasLayer': hub.KerasLayer})


#dictionary to label all the CIFAR-10 dataset classes.
classes = { 
    1: 'pink primrose',
    2: 'hard-leaved pocket orchid',
    3: 'canterbury bells',
    4: 'sweet pea',
    5: 'english marigold',
    6: 'tiger lily',
    7: 'moon orchid',
    8: 'bird of paradise',
    9: 'monkshood',
    10: 'globe thistle',
    11: 'snapdragon',
    12: 'colt\'s foot',
    13: 'king protea',
    14: 'spear thistle	',
    15: 'yellow iris',
    16: 'globe-flower',
    17: 'purple coneflower',
    18: 'peruvian lily',
    19: 'balloon flower',
    20: 'giant white arum lily',
    21: 'fire lily',
    22: 'pincushion flower',
    23: 'fritillary',
    24: 'red ginger',
    25: 'grape hyacinth',
    26: 'corn poppy',
    27: 'prince of wales feathers',
    28: 'stemless gentian',
    29: 'artichoke',
    30: 'sweet william',
    31: 'carnation',
    32: 'garden phlox',
    33: 'love in the mist',
    34: 'mexican aster',
    35: 'alpine sea holly',    
    36: 'ruby-lipped cattleya',
    37: 'cape flower',
    38: 'great masterwort',
    39: 'siam tulip',
    40: 'lenten rose',
    41: 'barbeton daisy',
    42: 'daffodil',
    43: 'sword lily',
    44: 'poinsettia',
    45: 'bolero deep blue',
    46: 'wallflower',
    47: 'marigold',
    48: 'buttercup',
    49: 'oxeye daisy	',
    50: 'common dandelion',
    51: 'petunia',
    52: 'wild pansy',
    53: 'primula',
    54: 'sunflower',
    55: 'pelargonium',
    56: 'bishop of llandaff',
    57: 'gaura',
    58: 'geranium',
    59: 'orange dahlia	',
    60: 'pink-yellow dahlia',
    61: 'cautleya spicata',
    62: 'japanese anemone',
    63: 'black-eyed susan',
    64: 'silverbush',
    65: 'californian poppy',
    66: 'osteospermum',
    67: 'spring crocus',
    68: 'bearded iris',
    69: 'windflower',
    70: 'tree poppy',
    71: 'gazania',
    72: 'azalea',
    73: 'water lily',
    74: 'rose',
    75: 'thorn apple',
    76: 'morning glory	',
    77: 'passion flower',
    78: 'lotus',
    79: 'toad lily',
    80: 'anthurium',
    81: 'frangipani',
    82: 'clematis',
    83: 'hibiscus',
    84: 'columbine',
    85: 'desert-rose',
    86: 'tree mallow',
    87: 'magnolia',
    88: 'cyclamen',
    89: 'watercress',
    90: 'canna lily',
    91: 'hippeastrum',
    92: 'bee balm',
    93: 'ball moss',
    94: 'foxglove',
    95: 'bougainvillea',
    96: 'camellia',
    97: 'mallow',
    98: 'mexican petunia',
    99: 'bromelia',
    100: 'blanket flower',
    101: 'trumpet creeper',
    102: 'blackberry lily',
}


#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification CIFAR10')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


def classify(file_path):
    #global label_packed

    image = cv2.imread(file_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32')
    image = image / 255
    #cv2.imshow('image', image)

    image = image.reshape((1, 224, 224, 3))
    pred = model.predict_classes(image)[0]
    sign = classes[pred+1]
    text = sign + '  ' + str(pred)
    print(sign)
    label.configure(foreground='#011638', text=text) 


def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image", command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

    
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

    
upload=Button(top,text="Upload an image",command=upload_image, padx=10,pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification CIFAR10",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
