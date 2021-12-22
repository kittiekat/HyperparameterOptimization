import os
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers


class NewImgProcess:
    def __init__(self, img_dir='', test_train_split=False, train_dir='', test_dir='', k=1, split = .75, img_height=224, img_width=224):
        """ 
        This class processes images for CNN training
        
        Inputs: 
        directories, string, for all/training/testing data
        test_train_split, True/False, indicates whether data is split into Test/Training directories or not. Use True if providing test_dir and train_dir
        k, integer, number of folds to use for K-Fold Validation
        split, float, percentage of data to use for training if data is not already split up
        img_height and img_width, integer, dimensions for resizing images
        
        Output:
        k-folds of 4D arrays
        """
        
        print("Processing Images...")
        
        self.k = k
        self.split = split
        self.testdata = {}
        self.traindata = {}
        self.img_height = img_height
        self.img_width = img_width
        self.img_shape = (self.img_height, self.img_width, 3) 
        
        self.tt_split = test_train_split
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.img_dir = img_dir
        
        self.count_all = 0
        self.count_test = 0
        self.count_train = 0 
        self.build_folds(self.split)
        
        if len([i for i in [img_dir, test_dir, train_dir] if i]) == 0:
            print("No images were processed")
        
                                            
    def build_folds(self, split):
         if test_train_split == True:
            self.imgs_from_directory(train_dir, df) 
            self.imgs_from_directory(test_dir, df)
                                            
            self.count_train = sum([len(files) for r, d, files in os.walk(train_dir)])
            self.count_test = sum([len(files) for r, d, files in os.walk(test_dir)])
            
            # Mobilenet preprocessing function scales the values to -1 to 1 range (not needed)
            self.gen_test = image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
            self.traindata[0] = self.gen_test.flow_from_directory(directory=self.train_dir, target_size=(self.img_height, self.img_width))

            self.gen_train = image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
            self.testdata[0] = self.gen_train.flow_from_directory(directory=self.test_dir, target_size=(self.img_height, self.img_width), shuffle=False)

            self.classes = list(self.testdata.class_indices.keys())
            self.class_count = len(self.classes)
                                              
                                            
        else: 
            file_names, class_labels = self.imgs_from_directory(all_dir)
            self.count_all = int(len(file_names)
            self.count_train = self.count_all * split)
            self.count_test = self.count_all - count_train
                
            for i in range(k):
                test_names, test_truth, train_names, train_truth = [],[],[],[]

                y = rnd.sample(range(len(file_names)), count_train)

                for i in range(len(file_names)):
                    if i in y: 
                        train_names.append(file_names[i])
                        train_truth.append(class_labels[i])
                    else:
                        test_names.append(file_names[i])
                        test_truth.append(class_labels[i])
                              
                # Create tf dataset       
                ds_train = tf.data.Dataset.from_tensor_slices((train_names, train_truth))                
                ds_test = tf.data.Dataset.from_tensor_slices((test_names, test_truth))
                
                # Resize, Shuffle and Batch
                
                
                # Process images in dataset
                self.testdata[i] = ds_train
                self.traindata[i] = ds_test                                  
    
    def imgs_from_directory(self, directory):
        file_names = []; class_labels = []
        sub_dirs = next(os.walk(directory))[1]
        sub_dirs.sort()

        for label in range(len(sub_dirs)):
            path = directory+'/'+sub_dirs[label]
            files = (file for file in os.listdir(path) 
                     if os.path.isfile(os.path.join(path, file)))
    
            dir_file_names= [path+'/'+file for file in files]
            dir_class_labels = [label]*len(dir_file_names)
            
            file_names+=dir_file_names
            class_labels+= dir_class_labels
       
        return file_names, class_labels
    
    def process_image_3D(self, path):
        """ This function processes the image from the provided file path and outputs a 3D array """
        img = image.load_img(path, target_size=(self.img_height, self.img_width)) # PIL Image
        img_array_3D = image.img_to_array(img) # (224, 224, 3) 3D Numpy Array
        return img_array_3D
    
    def process_image_4D(self, path):
        """ This function processes the image from the provided file path and outputs a 4D array """
        img_array_3D = self.process_image_3D(path)
        img_array_4D = np.expand_dims(img_array_3D, axis=0) # (1, 224, 224, 3) 4D Numpy Array
        out = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
        return img_array_4D
    


class ImgProcess:
    def __init__(self, train_dir, test_dir, k=1):
        """ This class processes images for a MobileNet CNN architecture only """
        """ Inputs: directories for training/testing/validation data (string) """
        print("Processing Images...")
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_shape = (224, 224, 3) 

        self.count_train = sum([len(files) for r, d, files in os.walk(train_dir)])
        self.count_test = sum([len(files) for r, d, files in os.walk(test_dir)])
        
        # Mobilenet preprocessing function scales the values to -1 to 1 range (not needed)
        self.gen_test = image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
        self.traindata = self.gen_test.flow_from_directory(directory=self.train_dir, target_size=(224, 224))
        
        self.gen_train = image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
        self.testdata = self.gen_train.flow_from_directory(directory=self.test_dir, target_size=(224, 224), shuffle=False)

        self.classes = list(self.testdata.class_indices.keys())
        self.class_count = len(self.classes)
            
    def random_image(self, directory, n=1, view = False):
        """ Inputs: directory - where to look
                    n - how many images to select
                    view - will show the images and predictions if True 
            Output: 2 list of n length with image paths and true classification values
        """
        
        y_truth = ['']*n
        file_path = ['']*n
        
        for x in range(n):
            # select a sub_directory
            image_dir = rnd.choice(os.listdir(directory))
            y_truth[x] = image_dir.title()

            # Pick a file
            file_name = rnd.choice(os.listdir(directory+"/"+image_dir))

            # File name
            file_path[x] = directory+image_dir+"/"+file_name

        if view:
            rows = int(np.ceil(n/3))
            cols = 3

            for x in range(n):
                plt.subplot(rows, cols, x+1)
                plt.rcParams["figure.figsize"] = (15,10)
                img = image.load_img(file_path[x], target_size=(224, 224))
                plt.imshow(img)
                plt.axis('off')
                plt.title("True Classification: "+y_truth[x])

            plt.show()

        return file_path, y_truth  

    def process_image(self, address):
        """ This function processes the image and outputs a format suitable for MobileNet classifier """

        return 999
        

class ModelParams: 
    def __init__(self, dict_params, model_name):
        # self.solution = float_solution # Why is this even needed?
        # self.int_solution = [int(float_solution[i]) for i in range(len(float_solution))] #Why is this even needed?
        self.params = dict_params
        self.name = model_name

        # Saved Model
        self.history = ''
        self.val_loss = 0
        self.val_accuracy = 0
        self.weights = ''


class HPO:
    def __init__(self, data, y, popsize=10): # , i=5.ipynb, F=.5.ipynb, cr=.9):
        """ Inputs: data is processed image object, 
        y_key - dictionary of parameters and their possible values in format: {param_name: [1,2,3]} 
        """
        # Image data object
        self.data = data
        
        # Where all evaluated models will be stored
        self.all_models = {}
        
        # Dictionary object of all parameters and their possible values
        self.y_key = y
        
        # List of parameters being optimized (needed for metrics)
        self.op = [x for x in y if len(y[x])>1]
        
        # List of possible values for parameters being optimized (needed for bounds)
        self.y = [np.arange(len(y[x])) for x in y if len(y[x])>1]
        
        # Bounds for values being optimized
        self.bounds = [[0, len(i)] for i in self.y]
        
        # Number of possible solutions
        self.n = np.prod([len(y[i]) for i in y])
        print(f"There are {self.n} possible solutions")
        
        # Data frame where model metrics will go
        self.df = pd.DataFrame(columns = list(self.y_key.keys())+['name','val_loss','val_accuracy'])
        
        # Differential Evolution Parameters (may not be needed)
        #self.iter = i
        self.popsize = popsize
        #self.F = F
        #self.cr = cr
        
    def translate_floats(self, solution): 
        """ Input: float vector solution
            Output: Dictionary object of parameters """
        
        dict_of_params={}
        x_sol = 0
        bools = [1 if len(self.y_key[x])>1 else 0 for x in self.y_key]
        solution = [int(solution[i]) for i in range(len(solution))]
 
        for x_bl in range(len(bools)):
            parameter = list(self.y_key.keys())[x_bl]
            
            if bools[x_bl] > 0:
                value = self.y_key[parameter][solution[x_sol]]
                x_sol+=1
                
            else:
                value = self.y_key[parameter][0]
            
            dict_of_params[parameter] = value
        
        return dict_of_params
    
    def save_model(self, model, my_params):
        """ Determines if the model should be saved and does any cleanup needed """
        
        loss_vector = [self.all_models[key].val_loss for key in self.all_models]
        loss_vector.sort()
        max_loss = loss_vector[self.popsize-1]
        
        if len(loss_vector) < self.popsize or my_params.val_loss < max_loss:
            if len(loss_vector) > self.popsize: self.clean_up(max_loss)
            my_params.weights = 'optimization/HPO Model' + my_params.name + '.params'
            model.save_weights(my_params.weights)
            print("Model Weights Saved.")
        else:
            print("High Validation Loss. Model Weights Discarded.")
            
    def clean_up(self, val_loss):
        # figure out the weights of the 10th lowest valuation model and delete the 2 files
        pass
        
    def name_model(self, p): 
        """ Parameters to String """
        # There must be a better way....
        p=str(p)
        p=p.replace("{", "")
        p=p.replace(":", "")
        p=p.replace("}", "")
        p=p.replace("'", "")
        p=p.replace("[", "")
        p=p.replace("]", "")
        return p
    
    def get_df(self):
        """ Extracts model information to data frame """        
        self.df["val_accuracy"] = [self.all_models[key].val_accuracy for key in self.all_models.keys()]
        self.df["val_loss"] = [self.all_models[key].val_loss for key in self.all_models.keys()]
        self.df["name"] = [self.all_models[key].name for key in self.all_models.keys()]

        for key in self.y_key.keys():
            self.df[key] =[self.all_models[i].params[key] for i in self.all_models.keys()]
            
    def export_to_csv(self, directory, fname):
        f_name=directory+fname+'-df.csv'
        self.df.to_csv(f_name)
        return f_name
    
    def all_model_metrics(self,all_metrics=True):
        """ Visualization of model metrics (must have more than one model """
        self.get_df()
        
        print(len(self.df),"Solutions evaluated out of", self.n,'possible solutions')
        
        # Frequency Graphs
        for i, col in enumerate(self.op):
            plt.figure(i)
            sns.countplot(x=col, data=self.df)

        # Violins
        for i, col in enumerate(self.op):
            plt.figure(i+5)
            sns.violinplot(x=col, y="val_accuracy", data=self.df)

    def evaluate_model(self, solution, weights=''):   
        """ Input: Vector of Float values """
        
        p = self.translate_floats(solution)
        new = ModelParams(p , self.name_model(p))

        # Check if solution has already been evaluated
        if str(new.name) not in self.all_models:
            if weights == '':
                self.all_models[new.name] = self.fit_model(new, self.data)
                #new.val_loss = rnd.random() # FOR TESTING
                #new.val_accuracy = 1- new.val_loss # FOR TESTING
                
            else:
                self.all_models[new.name] = self.fit_model(new, self.data, True ,weights)
                
            self.all_models[new.name]= new
            print(len(self.all_models), "have been evaluated")

        return self.all_models[new.name].val_loss

    def fit_model(self, my_params, img_obj, pretrain=False, weights_path='imagenet'):
        print("Fitting Model to:", my_params.name)
        
        # Define Model
        model = keras.models.Sequential()
        
        # To scale values to -1 to 1 range (which is what MobileNet was trained on)
        model.add(keras.layers.Rescaling(1./127.5, offset=-1))
            
        if pretrain:
            # Model Pretrained with ImageNet Weights must have Depth = 1
            # Reference: https://github.com/keras-team/keras/blob/v2.7.0/keras/applications/mobilenet.py
            # Model Top: global_average_pooling2d, reshape, dropout, conv2d, reshape and predictions
            # Alpha can be `0.25`, `0.50`, `0.75` or `1.0` only.
            # Model has 86 layers: 4 layers, followed with 13 blocks of 6-7 layers per block

            load_model = keras.applications.MobileNet(include_top=False,
                                                      weights=weights_path,
                                                      input_shape=img_obj.img_shape,
                                                      alpha=my_params.params['alpha'],
                                                      depth_multiplier=my_params.params['depth_multiplier'])

            # Blocks 10-13 -> 25, Blocks 11-13 -> 19, Blocks 12-13 -> 13
            for layer in load_model.layers[:-my_params.params['trainable_layers']]:
                layer.trainable = False
                
            model.add(load_model)
            model.add(keras.layers.GlobalAveragePooling2D(keepdims=True)) # 4D->2D, Max pooling can also be used 
            model.add(keras.layers.Dropout(my_params.params['dropout'], name='dropout'))
            model.add(keras.layers.Conv2D(img_obj.class_count, (1, 1), padding='same', name='conv_preds'))
            model.add(keras.layers.Reshape((img_obj.class_count,), name='reshape_2'))
            model.add(keras.layers.Activation(activation='softmax', name='predictions'))
        
        else:
            # Fully Trained Model 
            model.add(keras.applications.MobileNet(include_top=True,
                                                 weights=None,
                                                 input_shape=img_obj.img_shape,
                                                 classes=img_obj.class_count,
                                                 alpha=my_params.params['alpha'],
                                                 depth_multiplier=my_params.params['depth_multiplier'],
                                                 dropout=my_params.params['dropout']))

        lr = my_params.params['lr']
        epochs = my_params.params['epochs']
        steps_per_epoch = img_obj.count_train // my_params.params['batch_size']
        validation_steps = img_obj.count_test // my_params.params['batch_size']
        trainer = getattr(optimizers, my_params.params['optimizer'])(learning_rate=lr)

        model.compile(loss=keras.losses.CategoricalCrossentropy(),
                      optimizer=trainer,
                      metrics=['accuracy'])

        # Train the model
        model_history = model.fit(img_obj.traindata,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=img_obj.testdata,
                                  validation_steps=validation_steps,
                                  verbose=1)
        
        # Save the model
        my_params.history = model_history.history
        my_params.val_loss = model_history.history['val_loss'][-1]
        my_params.val_accuracy = model_history.history['val_accuracy'][-1]
        self.save_model(model, my_params)

        print("Model Metrics")
        print("Val Loss:", f'{my_params.val_loss:.4f}')
        print("Val Accuracy:", f"{my_params.val_accuracy:.4f}")

        return my_params


def plot_results(history, title):
    # Graph of Accuracy
    epochs = len(history['accuracy'])
    acc_train = history['accuracy']
    acc_val = history['val_accuracy']
    plt.plot(range(1, epochs + 1), acc_train, 'purple', label='Training accuracy')
    plt.plot(range(1, epochs + 1), acc_val, 'violet', label='Validation accuracy')
    plt.title(title + ' Training and Validation Accuracy')
    plt.ylim(0, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Graph of Loss
    loss_train = history['loss']
    loss_val = history['val_loss']
    plt.plot(range(1, epochs + 1), loss_train, 'purple', label='Training loss')
    plt.plot(range(1, epochs + 1), loss_val, 'violet', label='Validation loss')
    plt.title(title + ' Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
