import gradio as gr
import tensorflow as tf 
import numpy as np
import torchvision

model = tf.keras.models.load_model('InceptionV3.h5')
image = gr.input.image(shape=(224,224))
label = gr.output.Label(num_top_classes = 3)
imagenet_classes = torchvision.models.imagenet_classes
def predict(image):
    image = tf.image.resize(
        image,
        (224,224)
    )
    image = image/255
    prediction = model.predict(image)
    top_3_classes= np.argsort(prediction)[0][-3:]
    top_3_labels = [imagenet_classes[i] for i in top_3_classes]
    top_3_confidences = prediction[0][top_3_classes]
    return{
        
        'classes':top_3_classes,
        'labels' :top_3_labels,
        'confidences':top_3_confidences
    }
    
app = gr.Interface(fn =predict,inputs = image,outputs=label,title="Cotton Disease Prediction")
app.launch()