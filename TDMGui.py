# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:05:33 3022

@author: mohamed
"""

from tkinter import *
from tkinter import ttk
from tkinter import font
from tkinter import filedialog
from PIL import Image,ImageTk
import numpy as np
####### Image processing packages
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from scipy.spatial.distance import euclidean
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from skimage import feature
import cv2 
import mahotas
import numpy as np 
from imutils import paths
from imutils import contours
import os
import itertools
from tkinter import messagebox
 
 
#If you're working in another directory change this file path

#"C:/Users/mohamed/Downloads/TDM GUI/cracked_train" 


 
def histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def mean1(img):
    return img.mean()

def hog_descriptor(img):
    resized_img1 = resize(img, (128*4, 64*4))
    fd1= hog(resized_img1, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    return fd1

def LAB(img):
    LABimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([LABimg], [0], None, [256], [0, 256])
    return hist
 
def Zernike_moments(im):
    #preprocessing
    img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    img_len,img_width = img.shape[0],img.shape[1]
    centre = img_len//2,img_width//2
    seuil = img[centre[0]-2:centre[0]+2,centre[1]-2:centre[1]+2].mean()                     #img[img_dim[0]//2,img_dim[1]//2]
    couleur = 255
    options = [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
    result = cv2.threshold(img, int(seuil), couleur, options[0])[1]
    thresh = cv2.erode(result, None, iterations=5)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    features = mahotas.features.zernike_moments(thresh, cv2.minEnclosingCircle(cnts[0])[1], degree=8)
    
    return features

def hue_moment_descriptor(img):
    img_len,img_width = img.shape[0],img.shape[1]
    centre = img_len//2,img_width//2
    seuil = img[centre[0]-2:centre[0]+2,centre[1]-2:centre[1]+2].mean()                     #img[img_dim[0]//2,img_dim[1]//2]
    couleur = 255
    options = [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
    result = cv2.threshold(img, int(seuil), couleur, options[0])[1]
    # Calculate Moments 
    moments = cv2.moments(result) 
    # Calculate Hu Moments 
    huMoments = cv2.HuMoments(moments)
    return huMoments.reshape((7,))   

def display_imgs(results):
    SubZero = Toplevel() 
    SubZero.geometry("1280x720")
    main_frame = Frame(SubZero)
    main_frame.pack(fill=BOTH,expand=1)
    #Create a canvas
    my_canvas = Canvas(main_frame)
    my_canvas.pack(side=LEFT,fill=BOTH,expand=1)
    #Add scrollbar
    scrollV = Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
    scrollV.pack(side=RIGHT,fill=Y)
    #Config the canvas
    my_canvas.configure(yscrollcommand=scrollV.set)
    my_canvas.bind('<Configure>',lambda e:my_canvas.configure(scrollregion=my_canvas.bbox("all")))
    #Create canvas frame
    img_frame = Frame(my_canvas)
    my_canvas.create_window((0,0),window=img_frame,anchor="nw")
    col = 1
    row = 1
    for (i, (score, f)) in enumerate(results) :
        img = Image.open("C:/Users/mohamed/Downloads/TDM GUI/"+f)
        img = img.resize((300,300))
        img = ImageTk.PhotoImage(img)
        lab = Label(img_frame,image=img,text=f"Distance : {'%.2f' % score}",bg='#FFFFEF',fg='green',compound=TOP,highlightbackground='#30D5C8',highlightthickness=2,font=('bold',12))
        lab.grid(row=row,column=col)
        lab.image = img
        if(col==4):
            row = row +2
            col = 1
        else:
            col=col+1
    SubZero.mainloop()

def lbpdescriptor(image,numPoints=24,radius=8,eps=1e-7):
        # compute the Local Binary Pattern representation of the image, and then
        # use the LBP representation to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3),range=(0, numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

def texture_heralik_descriptor(img):
    return mahotas.features.haralick(img).mean(axis=0)



def standarize(mean,std,value):
    return (value-mean)/std +2 # because distance cant be negative



def search_color1(method):
    if method=='Histogram':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the imgss
        for imgsPath in paths.list_images(dataset):
            # load the imgs, convert it to grayscale, and describe it
            imgs = cv2.imread(imgsPath)
            gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            hist = histogram(gray)

            # update the index dictionary
            filename = imgsPath[imgsPath.rfind("/") + 1:]
            index[filename] = hist
        # load the query imgs and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = histogram(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))
        # show the query imgs and initialize the results dictionary
        results = {}
        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        # sort the results
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results   
    if method == 'Mean' :
        dataset = dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = mean1(gray)
            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = mean1(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features ,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results 
    if method == 'LAB' :
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = LAB(images)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = LAB(query)

        # show the query image and initialize the results dictionary
        results = {}
        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        return results  
        
def search_shape1(method):
    if method=='HOG method':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            try :
                images = cv2.imread(query)
                gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
                hist = hog_descriptor(images)

                # update the index dictionary
                filename = imagePath[imagePath.rfind("/") + 1:]
                #index["C:/Users/mohamed/Downloads/TDM GUI/"+filename] = hist
                index[filename] = hist
            except :
                continue

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = hog_descriptor(query)

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results 
        

    if method == 'HUE moments':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query = query  = import_file_path

        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = hue_moment_descriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = hue_moment_descriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results 

    if method=='Zernik method':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            
            try:
                hist = Zernike_moments(images)
            except:
                continue
            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = Zernike_moments(query)

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        results = standarize(results)
        # sort the results
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = sorted([(v, k) for (k, v) in results.items()])
        return results 
            
def search_texture1(method):
    if method == 'LBP':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path

        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = lbpdescriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = lbpdescriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results 
    if method== 'Haralik':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = texture_heralik_descriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = texture_heralik_descriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            results[k] = d
        # sort the results
        array = np.fromiter(results.values(), dtype=float)
        mean = array.mean()
        std = array.std()
        results = {k: standarize(mean,std,v) for k, v in results.items()}
        return results 





#The app's info
root = Tk()
root.title('Image Search Engine')
root.geometry("1280x720")
root.iconbitmap('C:/Users/mohamed/Downloads/TDM GUI/TDM.ico')
#root.configure(background="lightblue")
root.option_add('*tearOff', FALSE)
# Add image file
bg = PhotoImage(file = "C:/Users/mohamed/Downloads/TDM GUI/bgIm.png")
# Show image using label
label1 = Label( root, image = bg)
label1.place(x = 0, y = 0)
  
    
    
#Style    

# Add Menu
my_menu = Menu(root,background='blue',fg='white')
root.config(menu=my_menu)
#Search Menu
OptionMenu = Menu(my_menu, tearoff=0)
my_menu.add_cascade(label="Options",menu=OptionMenu)
# Drop down menu
OptionMenu.add_command(label="Themes")
OptionMenu.add_separator()
OptionMenu.add_command(label="About")

#Creating style for our comboBox
# Changing the combobox menu style.
root.option_add('*TCombobox*Listbox.selectBackground', '#30D5C8') # change highlight color
root.option_add('*TCombobox*Listbox.selectForeground', 'white') # change text color
style = ttk.Style()
style.configure('TCombobox', background='#30D5C8',bordercolor='#30D5C8') # Create a border around the combobox button.
style.map('TCombobox', foreground=[('hover', 'black')], background=[('hover', '#30D5C8')]) # style the combobox                                                                                           


ImageFrame = Frame(root,highlightbackground='#30D5C8',highlightthickness=3,height=10,width=100)
ImageFrame.pack(padx=10,pady=30)

# two panes, each of which would get widgets gridded into it:
f1 = Frame(root,highlightbackground='#30D5C8',highlightthickness=3,width=200,height=400)
label11 = Label(f1,text='Commands')
f2 = Frame(root,highlightbackground='#30D5C8',highlightthickness=3,width=350,height=400) 
label12 = Label(f2,text='Image')

f1.pack(fill=BOTH, expand=True,side=LEFT,padx=5,pady=15)
f2.pack(fill=BOTH,expand=True,side=LEFT,padx=5,pady=15)

#Inserting the logo
image = Image.open('C:/Users/mohamed/Downloads/TDM GUI/imglogo.png')
photo = ImageTk.PhotoImage(image)

label = Label(f1, image = photo)
label.image = photo
label.grid(row=7,column=5)


def defocus(event):
    event.widget.master.focus_set()

# Add Combobox & Percentages
#Spin 1
spinvalColor = StringVar()
s = ttk.Spinbox(f1, from_=0.0, to=100.0,increment=25,textvariable=spinvalColor)
s.grid(row=1,column=2)
s.configure(width=5,font=('calibri',12))
#Spin 2
spinvalShape = StringVar()
s1 = ttk.Spinbox(f1, from_=0.0, to=100.0,increment=25,textvariable=spinvalShape)
s1.grid(row=3,column=2)
s1.configure(width=5,font=('calibri',12))
#Spin 3
spinvalText = StringVar()
s2 = ttk.Spinbox(f1, from_=0.0, to=100.0,increment=25,textvariable=spinvalText)
s2.grid(row=5,column=2)
s2.configure(width=5,font=('calibri',12))
# Add % Label
percentLabel1 = Label(f1,
              text="%",
  
              # Changing font-size here
              font=("Segoe Script", 18),
              )
percentLabel2 = Label(f1,
              text="%",
  
              # Changing font-size here
              font=("Segoe Script", 18),
              )
percentLabel3 = Label(f1,
              text="%",
  
              # Changing font-size here
              font=("Segoe Script", 18),
              )



#Color methods
colorVar = StringVar()
colMeth = ttk.Combobox(f1, textvariable=colorVar)
ColorLabel = Label(f1,
              text="Color",
  
              # Changing font-size here
              font=("Segoe Script", 18),
              )
ColorLabel.grid(row=1,column=0,padx=10,pady=10)
colMeth.bind('<FocusIn>', defocus)
colMeth['values'] = ('Mean', 'Histogram', 'LAB')
colMeth.state(["readonly"])
colMeth.config(font=('calibri', '14'))
colMeth.grid(row=1,column=1,padx=10,pady=30)
percentLabel1.grid(row=1,column=3,padx=10,pady=10)


#Shape

shapeVar = StringVar()
shapeMeth = ttk.Combobox(f1, textvariable=shapeVar)
shapeLabel = Label(f1,
              text="Shape",
  
              # Changing font-size here
              font=("Segoe Script", 18),
              )
shapeLabel.grid(row=3,column=0,padx=10,pady=10)
shapeMeth.bind('<FocusIn>', defocus)
shapeMeth['values'] = ('HOG method', 'HUE moments', 'Zernik method')
shapeMeth.state(["readonly"])
shapeMeth.config(font=('calibri', '14'))
shapeMeth.grid(row=3,column=1,padx=10,pady=30)
percentLabel2.grid(row=3,column=3,padx=10,pady=10)


#Texture
textVar = StringVar()
textMeth = ttk.Combobox(f1, textvariable=textVar)
textLabel = Label(f1,
              text="Texture",
              # Changing font-size here
              font=("Segoe Script", 18),
              )
textLabel.grid(row=5,column=0,padx=9,pady=10)
textMeth.bind('<FocusIn>', defocus)
textMeth['values'] = ('LBP', 'Haralik')
textMeth.state(["readonly"])
textMeth.config(font=('calibri', '14'))
textMeth.grid(row=5,column=1,padx=10,pady=30)
percentLabel3.grid(row=5,column=3,padx=10,pady=10)

def my_command():
    global image,import_file_path
    for widget in f2.winfo_children():
        widget.destroy() 
    import_file_path = filedialog.askopenfilename()
    image=Image.open(import_file_path)
    my_image = ImageTk.PhotoImage(image)
    label=Label(f2, image=my_image)
    label.image = my_image
    label.grid()
    mean = np.mean(image)
    std = str(np.std(image))
    
    print(mean)
def upload():
    return


#Import the image using PhotoImage function
click_btn= PhotoImage(file='C:/Users/mohamed/Downloads/TDM GUI/upload2.png')

#Let us create a label for button event
img_label= Label(image=click_btn)

button= Button(ImageFrame, image=click_btn,command=my_command ,bd=0)
button.grid(row=1,column=0,padx=10,pady=5)

PathEntry = Entry(ImageFrame)
PathEntry.configure(highlightbackground='#30D5C8',highlightthickness=2,width=30,font=('calibri','20'))
PathEntry.grid(row=1,column=3,padx=10,pady=5)
############################ Add grey text on entry box ##############################################
def on_entry_click(event):
    """function that gets called whenever entry is clicked"""
    if PathEntry.get() == 'Enter your image path...':
       PathEntry.delete(0, "end") # delete all the text in the entry
       PathEntry.insert(0, '') #Insert blank for user input
       PathEntry.config(fg = 'black')
def on_focusout(event):
    if PathEntry.get() == '':
        PathEntry.insert(0, 'Enter your image path...')
        PathEntry.config(fg = 'grey')

################################ Remove focus from frames ###############################################


# Add an upload button
def upload_btn_method():
    global image,import_file_path
    for widget in f2.winfo_children():
        widget.destroy() 
    import_file_path = PathEntry.get()
    image=Image.open(import_file_path)
    my_image = ImageTk.PhotoImage(image)
    label=Label(f2, image=my_image)
    label.image = my_image
    label.grid()


upload_btn= PhotoImage(file='C:/Users/mohamed/Downloads/TDM GUI/button_upload.png')

#Let us create a label for button event
img_label2= Label(image=upload_btn)

upload_button= Button(ImageFrame, image=upload_btn,command=upload_btn_method,borderwidth=0)
upload_button.grid(row=1,column=4,padx=10,pady=5)


PathEntry.insert(0, 'Enter your image path...')
PathEntry.bind('<FocusIn>', on_entry_click)
PathEntry.bind('<FocusOut>', on_focusout)
PathEntry.config(fg = 'grey')

#Hybrid search method
def hybrid_search():
    per1 = int(spinvalColor.get())/100
    per2 = int(spinvalShape.get())/100
    per3 = int(spinvalText.get())/100
    print(per1,"|",per2,"|",per3)
    method1 = colorVar.get()
    method2 = shapeVar.get()
    method3 = textVar.get()
    if(per1+per2+per3 > 1):
        messagebox.showinfo("Invalid sum of percentages!!","Please check your percentages & try again")
    else :
        results_color = search_color1(method1)
        results_color.update((key, value * (1-per1)) for key, value in results_color.items())
        results_shape = search_shape1(method2)
        results_shape.update((key, value * (1-per2)) for key, value in results_shape.items())
        results_texture = search_texture1(method3)
        results_texture.update((key, value * (1-per1)) for key, value in results_texture.items())
        results=dict(itertools.chain(results_color.items(),results_shape.items()))
        results=dict(itertools.chain(results.items(),results_texture.items()))
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)

def search_color():
    if colorVar.get()=='Histogram':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        index = {}
        # loop over the imgss
        for imgsPath in paths.list_images(dataset):
            # load the imgs, convert it to grayscale, and describe it
            imgs = cv2.imread(imgsPath)
            gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            hist = histogram(gray)

            # update the index dictionary
            filename = imgsPath[imgsPath.rfind("/") + 1:]
            index[filename] = hist
        # load the query imgs and extract Local Binary Patterns from it
        query = cv2.imread(import_file_path)
        queryFeatures = histogram(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))
        # show the query imgs and initialize the results dictionary
        results = {}
        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])[:10]
        display_imgs(results)   
    if colorVar.get() == 'Mean' :
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = mean1(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(import_file_path)
        queryFeatures = mean1(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features ,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)
    if colorVar.get() == 'LAB' :
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = LAB(images)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = LAB(query)

        # show the query image and initialize the results dictionary
        results = {}
        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)
        
def search_shape():
    if shapeVar.get()=='HOG method':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            try :
                images = cv2.imread(imagePath)
                gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
                hist = hog_descriptor(images)

                # update the index dictionary
                filename = imagePath[imagePath.rfind("/") + 1:]
                #index["C:/Users/mohamed/Downloads/TDM GUI/"+filename] = hist
                index[filename] = hist
            except :
                continue

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(import_file_path)
        queryFeatures = hog_descriptor(query)

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        
        display_imgs(results)

    if shapeVar.get() == 'HUE moments':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query = import_file_path

        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = hue_moment_descriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = hue_moment_descriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)

    if shapeVar.get()=='Zernik method':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            
            try:
                hist = Zernike_moments(images)
            except:
                continue
            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = Zernike_moments(query)

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            #d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)  
            
def search_texture():
    if textVar.get() == 'LBP':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query  = import_file_path

        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = lbpdescriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = lbpdescriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary

        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)
    if textVar.get() == 'Haralik':
        dataset = "C:/Users/mohamed/Downloads/TDM GUI/cracked_train"
        query = import_file_path
        index = {}
        # loop over the images
        for imagePath in paths.list_images(dataset):
            # load the image, convert it to grayscale, and describe it
            images = cv2.imread(imagePath)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            hist = texture_heralik_descriptor(gray)

            # update the index dictionary
            filename = imagePath[imagePath.rfind("/") + 1:]
            index[filename] = hist

        # load the query image and extract Local Binary Patterns from it
        query = cv2.imread(query)
        queryFeatures = texture_heralik_descriptor(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

        # show the query image and initialize the results dictionary
        results = {}

        # loop over the index
        for (k, features) in index.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = euclidean(features,queryFeatures)
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])
        display_imgs(results)
            
        

hybrid_btn= PhotoImage(file='C:/Users/mohamed/Downloads/TDM GUI/button_hybrid.png')

#Let us create a label for button event
img_label1= Label(image=hybrid_btn)

button= Button(f1, image=hybrid_btn,command=hybrid_search,bd=0)
button.grid(row=7,column=1,padx=10,pady=10)

#Add individual search buttons
search_btn= PhotoImage(file='C:/Users/mohamed/Downloads/TDM GUI/button_search.png')
close_btn= PhotoImage(file='C:/Users/mohamed/Downloads/TDM GUI/button_x.png')

#Let us create a label for button event
img_label2= Label(image=hybrid_btn)

button1= Button(f1, image=search_btn,command=search_color,bd=0)
button2= Button(f1, image=search_btn,command=search_shape,bd=0)
button3= Button(f1, image=search_btn,command=search_texture,bd=0)
button1.grid(row=1 ,column=5,padx=10,pady=10)
button2.grid(row=3 ,column=5,padx=10,pady=10)
button3.grid(row=5 ,column=5,padx=10,pady=10)



root.mainloop()