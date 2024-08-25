import os
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import streamlit as st


# Define constants
MODEL_PATH = "PlantVillage.keras"
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

# Disease solutions
SOLUTIONS = {
    "Tomato_Bacterial_spot": "Fertilizers:\n1. Bonide Citrus, Fruit & Nut Orchard Spray (32 Oz)\n2. Bonide Infuse Systemic Fungicide...\n3. Hi-Yield Captan 50W fungicide (1...\n4. Monterey Neem Oil",
    "Tomato_Early_blight": "1. Mancozeb Flowable with Zinc Fungicide Concentrate\n2. Spectracide Immunox Multi-Purpose Fungicide Spray Concentrate For Gardens\n3. Southern Ag – Liquid Copper Fungicide\n4. Bonide 811 Copper 4E Fungicide\n5. Daconil Fungicide Concentrate.",
    "Tomato_healthy": "Your Plant Is Healthier.",
    "Tomato_Late_blight": "Plant resistant cultivars when available.\nRemove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.\nWater in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation.\nDestroy all tomato and potato debris after harvest.",
    "Tomato_Leaf_Mold": "Fungicides:\n1. Difenoconazole and Cyprodinil\n2. Difenoconazole and Mandipropamid\n3. Cymoxanil and Famoxadone\n4. Azoxystrobin and Difenoconazole",
    "Tomato_Septoria_leaf_spot": "Use disease-free seed and don't save seeds of infected plants\nStart with a clean garden by disposing all affected plants.\nWater aids the spread of Septoria leaf spot. Keep it off the leaves as much as possible by watering at the base of the plant only.\nProvide room for air circulation. Leave some space between your tomato plants so there is good airflow.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Prune leaves, stems, and other infested parts of plants well past any webbing and discard in trash (and not in compost piles). Don’t be hesitant to pull entire plants to prevent the mites from spreading to its neighbors.\nUse the Bug Blaster to wash plants with a strong stream of water and reduce pest numbers.\nCommercially available beneficial insects, such as ladybugs, lacewing, and predatory mites are important natural enemies. For best results, make releases when pest levels are low to medium.\nDust on leaves, branches, and fruit encourages mites. A mid-season hosing (or two!) to remove dust from trees is a worthwhile preventative.\nInsecticidal soap or botanical insecticides can be used to spot treat heavily infested areas.",
    "Tomato__Target_Spot": "1. Remove old plant debris at the end of the growing season; otherwise, the spores will travel from debris to newly planted tomatoes in the following growing fc, thus beginning the disease anew. Dispose of the debris properly and don’t place it on your compost pile unless you’re sure your compost gets hot enough to kill the spores.\n2. Rotate crops and don’t plant tomatoes in areas where other disease-prone plants have been located in the past year – primarily eggplant, peppers, potatoes or, of course – tomatoes. Rutgers University Extension recommends a three-year rotation cycle to reduce soil-borne fungi.\n3. Pay careful attention to air circulation, as target spot of tomato thrives in humid conditions. Grow the plants in full sunlight. Be sure the plants aren’t crowded and that each tomato has plenty of air circulation. Cage or stake tomato plants to keep the plants above the soil.\n4. Water tomato plants in the morning so the leaves have time to dry. Water at the base of the plant or use a soaker hose or drip system to keep the leaves dry. Apply a mulch to keep the fruit from coming in direct contact with the soil. Limit mulch to 3 inches or less if your plants are bothered by slugs or snails.\n5. You can also apply fungal spray as a preventive measure early in the season, or as soon as the disease is noticed.",
    "Tomato__Tomato_mosaic_virus": "Fungicides will not treat this viral disease.\nAvoid working in the garden during damp conditions (viruses are easily spread when plants are wet).\nFrequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc.\nRemove and destroy all infected plants. Do not compost.\nDo not save seed from infected crops.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers.\nCover plants with floating row covers of fine mesh (Agryl or Agribon) to protect from whitefly infestations.\nPractice good weed management in and around fields to the extent feasible.\nRemove and destroy old crop residue and volunteers on a regional basis."
}

# Load the model once
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"The error {e}")
    #print(model.input_shape)

def detect(image_path):
    ht = 50
    wd = 50

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(wd, ht))
    if img is None:
    
        raise ValueError(f"Image not found or unable to read the image: {image_path}")
 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    x = np.array(img_array, 'float32')
    x = x/255
    predictions = model.predict(x)[0]
    # Predict the disease
    #predictions = model.predict(image)[0]
    max_prob = max(predictions)
    max_index = predictions.tolist().index(max_prob)
    label = CLASS_NAMES[max_index]
    percentage = float("{0:.2f}".format(max_prob * 100))

    return label, percentage

# Streamlit application
st.title("Leaf Disease Detection and Classification")

uploaded_file = st.file_uploader("Choose an image...", 
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False)
print(uploaded_file)
if uploaded_file is not None:
    # Save the uploaded file
    image_path = os.path.join("static/", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Detect the disease
        result, percentage = detect(image_path)
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence level:** {percentage}%")

        # Show the solution
        solution = SOLUTIONS.get(result, "No solution found.")
        st.write("**Solution:**")
        st.write(solution)
    except Exception as e:
        st.error(f"Error during detection: {e}")
