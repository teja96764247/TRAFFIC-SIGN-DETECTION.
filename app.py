import streamlit as st
import tensorflow as tf



@st.cache_data
def load_model():
    model=tf.keras.models.load_model('traffic.hdf5')
    return model
model=load_model()

st.write("""
       #TRAFFIC SIGN IDENTIFICATION
         """
)

file=st.file_uploader("Please upload an Traffic Sign mage",type=['jpg','png'])

from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(64,64)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,64,64,3])
    prediction=model.predict(img_reshape)
    output=np.argmax(prediction)
    if(output==0):
        return 'The Traffic Sign is COMPULSARY_KEEP_RIGHT'
    elif(output==1):
        return 'The Traffic Sign is CROSS_ROAD'
    elif(output==2):
        return('The Traffic Sign is Compulsory Keep BothSide')
    elif(output==3):
        return('The Traffic Sign is Compulsory Keep Right')
    elif(output==4):
        return('The Traffic Sign is Cycle crossing')
    elif(output==5):
        return('The Traffic Sign is Danger')
    elif(output==6):
        return('The Traffic Sign is GAP_IN_MEDIAN')
    elif(output==7):
        return('The Traffic Sign is HORN_PROHIBITED')
    elif(output==8):
        return('The Traffic Sign is HUMP_OR_ROUGH_ROAD')
    elif(output==9):
        return('The Traffic Sign is LEFT_TURN_PROHIBITED')
    elif(output==10):
        return('The Traffic Sign is Maximum Speed 30')
    elif(output==11):
        return('The Traffic Sign is NO Stopping')
    elif(output==12):
        return('The Traffic Sign is NO_ENTRY')
    elif(output==13):
        return('The Traffic Sign is NO_STOPPING_OR_STANDING')
    elif(output==14):
        return('The Traffic Sign is No Entry')
    elif(output==15):
        return('The Traffic Sign is One way Traffic')
    elif(output==16):
        return('The Traffic Sign is PEDESTRIAN_CROSSING')
    elif(output==17):
        return('The Traffic Sign is Pedestrain')
    elif(output==18):
        return('The Traffic Sign is Right Margin')
    elif(output==19):
        return('The Traffic Sign is Right Turn Prohibited')
    elif(output==20):
        return('The Traffic Sign is Roundabouts')
    elif(output==21):
        return('The Traffic Sign is SPEED_LIMIT_30')
    elif(output==22):
        return('The Traffic Sign is SPEED_LIMIT_40')
    elif(output==23):
        return('The Traffic Sign is SPEED_LIMIT_50')
    elif(output==24):
        return('The Traffic Sign is SPEED_LIMIT_60')
    elif(output==25):
        return('The Traffic Sign is SPEED_LIMIT_70')
    elif(output==26):
        return('The Traffic Sign is SPEED_LIMIT_80')
    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    result=predict_function(image,model)
    st.success(result)