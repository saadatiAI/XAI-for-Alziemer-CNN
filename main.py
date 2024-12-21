import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

## load VeryMildDemented data

for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/VeryMildDemented/"):
    label = "VeryMildDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/VeryMildDemented/"):
    label = "VeryMildDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)

## load MildDemented data
images = []
labels = [] 
ImagePath_List=[]
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/MildDemented/"):
    label = "MildDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/MildDemented/"):
    label = "MildDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)


## load ModerateDemented data

for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/ModerateDemented/"):
    label = "ModerateDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/ModerateDemented/"):
    label = "ModerateDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)


## load NonDemented data
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/NonDemented/"):
    label = "NonDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)
for dir_path in glob.glob("./input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/NonDemented/"):
    label = "NonDemented"
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        labels.append(label)
        ImagePath_List.append(image_path)
images = np.array(images)
labels = np.array(labels)

model_cnn = load_model(r"model_cnn.h5", custom_objects = {"precision": precision, "recall": recall, "f1": f1})


class_names = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
wrong_preds_ids =[]
true_preds_ids=[]
predictions = model_cnn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(f"Real Class: \"{ class_names[np.argmax(y_test[image_id_for_explain])]}\"")
for i in range(len(predictions)):
    if(np.argmax(y_test[i]) != predicted_classes[i]):
        wrong_preds_ids.append(i)
    else:
        true_preds_ids.append(i)


print("Number of Wrong Predictions: ", len(wrong_preds_ids))
print("False Predictions: (image_id)", wrong_preds_ids)
print(f"Predicted Class: \"{ class_names[predicted_classes[image_id_for_explain]]}\"")




# =========================================================================================================================================
## shap method
# =========================================================================================================================================
import matplotlib.image as mpimg


def Run_SHAP(image_id_for_explain, max_evals_list=[ 1000, 3000,5000]):
    class_names = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
    print(class_names[np.argmax(y_test[image_id_for_explain])])
    def f(x):
        tmp = x.copy()
        return model_cnn.predict(tmp)
    masker_blur = shap.maskers.Image("blur(45,45)", X_test[image_id_for_explain].shape)
    explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

    for max_evals in max_evals_list:
        shap_values_fine = explainer_blur(X_test[image_id_for_explain:image_id_for_explain+1], max_evals=max_evals, batch_size=200, outputs=shap.Explanation.argsort.flip[:4])
        print(f" max evals = {max_evals}")
        shap.image_plot(shap_values_fine)



# =========================================================================================================================================
## lime method
# =========================================================================================================================================

def Run_LIME(image_id_for_explain):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_test[image_id_for_explain], model_cnn.predict, top_labels=4, hide_color=0, num_samples=100)
    class_names = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    for i, class_idx in enumerate(explanation.top_labels):
        # Get the image explanation for the current class
        temp, mask = explanation.get_image_and_mask(
            class_idx, positive_only=False, num_features=10, hide_rest=False
        )

        # Display the explanation for each class in a subplot
        ax[i].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ax[i].set_title(f"Explanation for {class_names[class_idx]}")  # Label with class name
        ax[i].axis('off')  # Hide axes for a cleaner look

    # Display the final grid of images
    plt.tight_layout()
    plt.show()



# =========================================================================================================================================
## grad cam
# =========================================================================================================================================
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 45, 45, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def GradCAM_heatmap(image_id_for_explain):
    # Prepare image
    img_path = test_image_paths[image_id_for_explain]
    img_array = get_img_array(img_path, size=(45,45))

    # Make model
    model = model_cnn

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    # preds = model.predict(img_array)
    # print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, "conv2d_5")

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    return heatmap
from IPython.display import Image, display
import matplotlib as mpl
def Run_GradCAM(image_id_for_explain, cam_path="cam.jpg", alpha=0.8):
    heatmap = GradCAM_heatmap(image_id_for_explain)
    img_path = test_image_paths[image_id_for_explain]
        # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Convert to grayscale to find the important region
    gray = np.mean(img, axis=-1)  # Convert to grayscale by averaging channels

    # Threshold to create a binary mask for the important region
    binary_mask = gray > 1  # Threshold for "non-black" pixels

    # Find coordinates of the bounding box
    coords = np.argwhere(binary_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image to the bounding box
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((cropped_img.shape[1], cropped_img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + cropped_img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    plt.imshow(mpimg.imread(cam_path))
    plt.show()


# =========================================================================================================================================
#Run Methods
# =========================================================================================================================================


image_id_for_explain = 236

def RealClass(image_id):
    return class_names[np.argmax(y_test[image_id])]

def PredictedClass(image_id):
    return class_names[predicted_classes[image_id]]

def pred_index(image_id):
    return predicted_classes[image_id]

print(pred_index(image_id_for_explain))

Run_GradCAM(image_id_for_explain, pred_index(image_id_for_explain))
Run_SHAP(image_id_for_explain,[5000])
Run_LIME(image_id_for_explain)
plt.imshow(cv2.imread(test_image_paths[image_id]))
plt.show()





# for image_id in true_preds_ids[:100]:
#     if PredictedClass(image_id)=="MildDemented":
#         print("================================================================================================")
#         print("image id: ", image_id)
#         plt.imshow(cv2.imread(test_image_paths[image_id]))
#         Run_GradCAM(image_id)
#         Run_SHAP(image_id,[2000])
#         Run_LIME(image_id) 
#         # Image(test_image_paths[image_id])
#         plt.show()

    
#     if Real_class(image_id)=="ModerateDemented":
#         print(image_id)

# print("Real Class, Predicted Class")

# for image_id in wrong_preds_ids:
#     print(f"{RealClass(image_id)}, {PredictedClass(image_id)}")

# for i in range(len(predictions)):
    # print(f"{RealClass(i)}, {PredictedClass(i)}")


# predict_dict = [number for number in range(1, 10)]
# for image_id in y_test:







