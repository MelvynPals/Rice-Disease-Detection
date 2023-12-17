import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sdfljsdf-dsfsdkl32fr'

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/mobile")
def mobile_home():
    return render_template('mobile-version.html')

model = torch.load("models/Beit-for-RD-model-final.pth")
model.eval()

@app.route("/predict", methods=['POST'])
def predict():
    image_file = request.files['image-file']
    if image_file and image_file.filename != '':
        filename = 'image.' + image_file.filename.rsplit('.', 1)[1].lower()
        image_file.save(filename)
        # Open and resize the image
        try:
            img = Image.open(filename)
            img = img.resize((224, 224))
            # Convert PNG to JPEG if necessary
            if filename.endswith('.png'):
                rgb_img = img.convert('RGB')
                rgb_img.save('image.jpg')
                filename = 'image.jpg'
            else:
                img.save(filename)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        # Handle the error, e.g. by returning an error message
        return jsonify({'predicted_class': 'No file uploaded'})

    image = Image.open(filename)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)
    _, predicted_class = torch.max(output.logits, 1)

    # Render the prediction result in an HTML template
    return jsonify({'predicted_class': predicted_class.item()})
    #return render_template('index.html', predicted_class=predicted_class.item())

if __name__=="__main__":
    app.debug=True
    app.run()