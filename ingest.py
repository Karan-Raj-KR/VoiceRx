from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
from dotenv import load_dotenv
import os, uuid

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

COLLECTION_NAME = "health_faqs"

faqs = [
    # Fever & General
    "Fever above 103°F lasting more than 3 days requires immediate medical attention.",
    "For mild fever, take paracetamol and rest. Stay hydrated with water and fluids.",
    "Common cold symptoms include runny nose, sore throat, and mild fever. Rest and fluids help.",

    # Chest Pain & Heart
    "Chest pain with shortness of breath is a medical emergency. Call 108 immediately.",
    "Heart attack symptoms include chest pain, left arm pain, and sweating. Call 108 immediately.",
    "Angina is chest discomfort caused by reduced blood flow to the heart and requires prompt evaluation by a cardiologist.",
    "Irregular heartbeat or palpitations that last more than a few minutes should be evaluated by a doctor.",
    "Maintaining a heart-healthy diet rich in fruits, vegetables, and whole grains reduces the risk of cardiovascular disease.",

    # Diabetes
    "Diabetes patients should monitor blood sugar daily and avoid sugary foods.",
    "Type 2 diabetes can often be managed with lifestyle changes including a balanced diet, regular exercise, and weight control.",
    "Hypoglycemia, or low blood sugar, causes shakiness, sweating, and confusion and should be treated immediately with glucose tablets or juice.",
    "Diabetic foot care is critical — inspect feet daily for sores or cuts that may not heal properly.",
    "HbA1c testing every 3 months helps diabetics track long-term blood sugar control.",
    "Uncontrolled diabetes can damage kidneys, eyes, and nerves over time, so consistent management is essential.",

    # Hypertension
    "High blood pressure can be managed with low salt diet, exercise, and prescribed medication.",
    "Blood pressure should be checked regularly. Normal reading is below 120/80 mmHg.",
    "Reducing sodium intake to less than 2,300 mg per day significantly lowers blood pressure.",
    "Chronic stress raises blood pressure, so relaxation techniques like deep breathing and yoga are beneficial for hypertensive patients.",
    "Hypertension often has no symptoms, which is why it is called the silent killer and regular monitoring is essential.",
    "Alcohol consumption should be limited to reduce blood pressure and improve overall cardiovascular health.",

    # Dengue
    "Dengue symptoms are high fever, severe headache, and joint pain. See a doctor immediately.",
    "Dengue fever is spread by the Aedes mosquito, which bites mainly during daylight hours.",
    "Patients with dengue should drink plenty of fluids and avoid aspirin or ibuprofen, which can worsen bleeding.",
    "A sudden drop in platelet count during dengue fever is a warning sign that requires hospitalization.",
    "Using mosquito nets, repellents, and eliminating stagnant water near your home prevents dengue transmission.",

    # Malaria
    "Malaria is caused by Plasmodium parasites transmitted through bites of infected female Anopheles mosquitoes.",
    "Malaria symptoms include cyclic fever, chills, sweating, and headache and require immediate blood test confirmation.",
    "Anti-malarial drugs like chloroquine or artemisinin-based combinations must be completed fully to prevent drug resistance.",
    "Sleeping under insecticide-treated bed nets is one of the most effective ways to prevent malaria.",
    "Pregnant women are especially vulnerable to malaria and should take prophylactic medication as prescribed by their doctor.",

    # Asthma
    "Asthma patients should always carry their inhaler and avoid dust and smoke.",
    "Asthma triggers include pollen, pet dander, cold air, and respiratory infections, and identifying your triggers helps control attacks.",
    "A peak flow meter can help asthma patients monitor lung function daily and detect early signs of worsening.",
    "Long-term asthma control requires daily use of controller inhalers even when symptoms are absent.",
    "During an asthma attack, use a rescue inhaler, stay calm, sit upright, and seek emergency care if breathing does not improve.",

    # Pregnancy
    "Pregnancy checkups should happen every month in the first 6 months, then every 2 weeks.",
    "Folic acid supplementation before and during early pregnancy reduces the risk of neural tube defects in the baby.",
    "Pregnant women should avoid alcohol, smoking, and unprescribed medications as they can harm fetal development.",
    "Iron and calcium supplements are commonly recommended during pregnancy to support the baby's growth and prevent anemia.",
    "Warning signs during pregnancy include heavy bleeding, severe headache, blurred vision, and reduced fetal movement, which require immediate medical attention.",
    "Gestational diabetes should be screened for between weeks 24 and 28 of pregnancy.",

    # Vaccinations
    "Vaccination schedule for children includes BCG, DPT, and MMR at specific age milestones.",
    "The flu vaccine is recommended annually for elderly people, pregnant women, and those with chronic illnesses.",
    "Adults should keep their tetanus booster up to date, with a shot recommended every 10 years.",
    "The HPV vaccine is most effective when given before the onset of sexual activity and protects against cervical cancer.",
    "COVID-19 vaccines significantly reduce the risk of severe illness and hospitalization.",
    "Hepatitis B vaccination is recommended for all newborns and unvaccinated adults at risk of exposure.",

    # Dehydration
    "Dehydration symptoms include dark urine, dizziness, and dry mouth. Drink ORS solution.",
    "Adults should aim to drink at least 8 glasses of water daily, and more during hot weather or physical activity.",
    "Severe dehydration from vomiting or diarrhea may require intravenous fluids administered in a hospital setting.",
    "Oral rehydration solution (ORS) is the most effective first-line treatment for dehydration caused by diarrhea.",
    "Caffeinated beverages and alcohol increase fluid loss and should not be used to treat dehydration.",

    # Mental Health
    "Persistent sadness, loss of interest, and fatigue lasting more than two weeks may indicate clinical depression requiring professional help.",
    "Anxiety disorders are treatable with a combination of therapy, lifestyle changes, and in some cases medication.",
    "Talking to a trusted friend, family member, or mental health professional is an important first step when feeling overwhelmed.",
    "Regular physical activity releases endorphins that naturally improve mood and reduce symptoms of anxiety and depression.",
    "Sleep deprivation worsens mental health conditions; adults should aim for 7 to 9 hours of quality sleep per night.",
    "Suicide risk warning signs include withdrawal from others, giving away possessions, and talking about hopelessness — take these seriously and seek help immediately.",

    # Skin Conditions
    "Eczema flare-ups can be managed by moisturizing regularly, avoiding harsh soaps, and identifying personal triggers.",
    "Psoriasis is a chronic autoimmune skin condition that causes red, scaly patches and requires dermatologist-guided treatment.",
    "Apply broad-spectrum sunscreen with SPF 30 or higher daily to protect against UV-induced skin damage and skin cancer.",
    "Acne can be treated with topical retinoids, benzoyl peroxide, or prescribed antibiotics depending on severity.",
    "Any mole that changes in size, shape, or color should be evaluated by a dermatologist for possible skin cancer.",

    # Eye Care
    "Adults should have a comprehensive eye exam at least every two years to detect conditions like glaucoma and macular degeneration early.",
    "The 20-20-20 rule helps reduce eye strain — every 20 minutes, look at something 20 feet away for 20 seconds.",
    "Diabetics should have annual retinal exams as diabetes can cause diabetic retinopathy leading to blindness if untreated.",
    "Wearing UV-protective sunglasses outdoors reduces the risk of cataracts and other UV-related eye damage.",
    "Sudden vision loss, eye pain, or flashes of light are emergency symptoms that require immediate ophthalmology evaluation.",

    # Dental Health
    "Brushing teeth twice daily with fluoride toothpaste and flossing once a day prevents cavities and gum disease.",
    "Regular dental checkups every 6 months allow early detection and treatment of tooth decay and periodontal disease.",
    "Gum disease (periodontitis) has been linked to heart disease and diabetes, making oral hygiene important for overall health.",
    "Avoid smoking and tobacco products as they significantly increase the risk of oral cancer and tooth loss.",
    "Children should begin dental visits by age one to establish healthy oral habits early.",

    # Nutrition
    "A balanced diet should include a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.",
    "Excess consumption of processed foods and added sugars contributes to obesity, diabetes, and cardiovascular disease.",
    "Vitamin D deficiency is common and can lead to bone weakness; sunlight exposure and dairy products help maintain adequate levels.",
    "Iron-rich foods like spinach, lentils, and red meat help prevent anemia, especially in women of reproductive age.",
    "Eating smaller, frequent meals helps maintain stable blood sugar levels and supports healthy metabolism.",

    # Emergency Symptoms
    "Sudden severe headache described as the worst of your life may indicate a brain aneurysm and requires emergency care.",
    "Difficulty speaking, facial drooping, or sudden arm weakness are signs of stroke — call 108 immediately.",
    "High fever with stiff neck and sensitivity to light may indicate meningitis, a life-threatening emergency.",
    "Severe allergic reactions (anaphylaxis) cause throat swelling and breathing difficulty and require an epinephrine injection and emergency care immediately.",
    "Loss of consciousness, even briefly, should always be evaluated by a doctor to rule out serious underlying causes.",

    # Appointment & Prescription (retained from original)
    "For appointment booking, please call the front desk between 9 AM and 5 PM.",
    "For prescription refills, contact your doctor at least 3 days before your medication runs out.",
]

# Create collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Embed and upload
embeddings = list(embedding_model.embed(faqs))

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings[i].tolist(),
        payload={"text": faqs[i]}
    )
    for i in range(len(faqs))
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} FAQs to Qdrant.")