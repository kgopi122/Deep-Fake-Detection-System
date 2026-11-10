# VeriTrue – Multi-Modal AI Deepfake & Image Authenticity Detection System

<p align="center">
  <strong>AI-Powered Image Authenticity and Deepfake Detection Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React-Frontend-blue?logo=react" />
  <img src="https://img.shields.io/badge/Java-17+-orange?logo=openjdk" />
  <img src="https://img.shields.io/badge/Spring%20Boot-3.x-brightgreen?logo=springboot" />
  <img src="https://img.shields.io/badge/Python-AI%20Pipeline-yellow?logo=python" />
  <img src="https://img.shields.io/badge/ONNX-Model%20Runtime-purple" />
  <img src="https://img.shields.io/badge/Deployment-Vercel%20%7C%20Render-blue" />
</p>

---

## Overview

**VeriTrue** is an end-to-end AI-powered image authenticity and deepfake detection platform designed to identify synthetic, manipulated, and AI-generated content (AIGC).

The system combines a modern React frontend with a backend inference engine powered by ONNX Runtime. It performs neural network inference directly within the Java Virtual Machine using native C++ bindings, reducing the overhead associated with external inference service calls.

The platform uses an optimized INT8-quantized ONNX model to reduce model size and memory consumption while maintaining an efficient image detection workflow.

VeriTrue also incorporates a heuristic forensic fallback mechanism to maintain service availability when the ONNX model is unavailable or inference fails.

The system is designed with lightweight deployment environments in mind, including cloud platforms with limited memory resources.

## Key Features

### 1. In-Process AI Inference
- Executes neural network inference directly inside the Java Virtual Machine.
- Uses ONNX Runtime Java with native C++ bindings.
- Avoids external HTTP calls to a separate inference service.
- Supports efficient image authenticity analysis.

### 2. INT8 Model Quantization
- Converts trained PyTorch model weights into ONNX format.
- Applies INT8 dynamic weight quantization.
- Uses QUInt8 quantization.
- Reduces model disk footprint by approximately 75%.
- Produces a lightweight model named `veritrue_int8.onnx`.

### 3. Dual-Engine Detection Architecture
- Primary detection through the ONNX Runtime inference engine.
- Heuristic forensic analysis as a fallback.
- Automatically switches to fallback analysis if the ONNX model is unavailable or inference fails.
- Prevents inference failures from unnecessarily interrupting the application.

### 4. Memory-Optimized Execution
- Optimized JVM memory configuration.
- Designed for cloud environments with limited memory.
- Uses lightweight model artifacts to reduce resource consumption.
- Supports deployment on environments with approximately 512 MB RAM.

### 5. Modern React Interface
- Responsive user interface.
- Image upload functionality.
- Dynamic real/fake probability breakdown.
- Detection confidence visualization.
- Forensic feedback metrics.
- Environment-aware backend API configuration.

---

## System Architecture

VeriTrue follows a decoupled frontend and backend architecture with an integrated AI inference engine.

```text
                 +----------------------+
                 |      User Browser    |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |    React Frontend    |
                 |    Vercel Hosting    |
                 +----------+-----------+
                            |
                     REST API Request
                            |
                            v
        +-------------------------------------------+
        |          Backend Application              |
        |                                           |
        |              REST Controller              |
        |                     |                     |
        |                     v                     |
        |              Analysis Service             |
        |                     |                     |
        |          +----------+----------+          |
        |          |                     |          |
        |          v                     v          |
        |    +-------------+     +----------------+ |
        |    | ONNX Runtime|     | Heuristic      | |
        |    | Java Engine |     | Forensics      | |
        |    +------+------+     +----------------+ |
        |           |                               |
        |           v                               |
        |    INT8 Quantized Model                   |
        |    veritrue_int8.onnx                    |
        +---------------------+---------------------+
                              |
                              v
                    +-------------------+
                    | Detection Result  |
                    | Real % / Fake %   |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | React Dashboard   |
                    +-------------------+
```

### Image Processing Workflow

1. The user uploads an image through the React frontend.
2. The frontend sends the image to the backend through a REST API.
3. The backend processes the uploaded image.
4. The image is resized to `224 × 224` pixels.
5. RGB rescaling and ImageNet normalization are applied.
6. The ONNX Runtime engine performs neural network inference.
7. If ONNX inference is unavailable or fails, the heuristic forensic engine provides fallback analysis.
8. The backend calculates the real and fake probability scores.
9. The frontend displays the detection result and confidence information.

---

## Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| React.js | User interface development |
| Vite / Create React App | Frontend tooling |
| JavaScript | Application logic |
| Axios / Native Fetch API | Backend communication |
| HTML5 | Page structure |
| CSS3 | Interface styling |
| Vercel | Frontend deployment |

### Backend

| Technology | Purpose |
|---|---|
| Java 17+ | Backend programming language |
| Spring Boot 3.x | Backend application framework |
| Apache Maven | Dependency management and build automation |
| ONNX Runtime Java | In-process AI inference |
| Native C++ Bindings | Optimized model execution |
| Render | Backend deployment |

### AI and Machine Learning

| Technology | Purpose |
|---|---|
| Python | AI model processing |
| PyTorch | Model definition and training |
| ONNX | Model interoperability |
| ONNX Runtime | Model execution |
| onnxruntime.quantization | Model quantization |
| NumPy | Numerical operations |
| Pillow | Image processing |

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model format | INT8 Quantized ONNX |
| Quantization type | QUInt8 |
| Input dimensions | `[1, 3, 224, 224]` |
| Input format | BCHW |
| Model file | `veritrue_int8.onnx` |
| Fake class output index | `1` |
| Primary inference engine | ONNX Runtime Java |

---

## Repository Structure

```text
Deep-Fake-Detection-System/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── ...
│   ├── package.json
│   └── ...
│
├── python-detector/
│   └── ... Python AI detection and model processing
│
├── veritrue-backend/
│   └── ... Backend application
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── package.json
├── package-lock.json
├── vercel.json
└── README.md
```

---

## Model Quantization and Export Workflow

VeriTrue uses a custom Python workflow to convert trained PyTorch model weights into an optimized INT8 ONNX model.

The conversion process is handled by the `export_quantize.py` script.

### Step 1: Install Python Dependencies

```bash
pip install torch onnx onnxruntime pillow numpy
```

### Step 2: Export and Quantize the Model

To convert a trained checkpoint explicitly:

```bash
python scripts/export_quantize.py --pt-path models/model.pt
```

Alternatively, allow the script to discover the model automatically:

```bash
python scripts/export_quantize.py
```

The script performs the following operations:

1. Validates the input checkpoint.
2. Exports the PyTorch model into FP32 ONNX format.
3. Applies QUInt8 dynamic weight quantization.
4. Generates the optimized INT8 ONNX model.
5. Performs a sample inference verification.
6. Validates the exported model output.

### Step 3: Expected Verification Output

```text
[INFO] Exporting PyTorch model to FP32 ONNX format...
[INFO] Quantizing ONNX model using QUInt8 dynamic weight quantization...
[INFO] Successfully generated INT8 ONNX model at:
backend/src/main/resources/models/veritrue_int8.onnx

[VERIFICATION] Running sample inference pass on exported model...
[VERIFICATION] Raw Logits: [-1.4021, 2.1584]
[VERIFICATION] Softmax Probabilities -> Real: 2.76%, Fake: 97.24%

[VERIFICATION] Model Sanity Check PASSED!
```

**Note:** The output above is an example verification log.

---

## Local Development Setup

### Prerequisites

Install the following software:

| Software | Version |
|---|---|
| Java JDK | 17 or higher |
| Node.js | 18.x or higher |
| Python | 3.10+ |
| Git | Latest stable version |
| Maven | Compatible with the project |

### 1. Clone the Repository

```bash
git clone https://github.com/kgopi122/Deep-Fake-Detection-System.git

cd Deep-Fake-Detection-System
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Verify that the quantized model exists at:

```text
src/main/resources/models/veritrue_int8.onnx
```

If the model is missing, follow the model quantization and export workflow described above.

Configure `src/main/resources/application.properties`:

```properties
server.port=8000

veritrue.onnx.enabled=true

veritrue.onnx.modelPath=classpath:models/veritrue_int8.onnx

veritrue.onnx.fakeIndex=1
```

Build the application:

```bash
./mvnw clean compile
```

Start the backend:

```bash
./mvnw spring-boot:run
```

Backend URL:

```text
http://localhost:8000
```

### 3. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## Environment Configuration

### Frontend Environment Variables

| Variable | Environment | Description | Example |
|---|---|---|---|
| `VITE_API_BASE_URL` | Local | Local backend address | `http://localhost:8000` |
| `VITE_API_BASE_URL` | Production | Deployed backend URL | `https://veritrue-backend.onrender.com` |

Do not append API paths or trailing slashes to the base URL unless required by the frontend configuration.

### Backend Application Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `veritrue.onnx.enabled` | Boolean | `true` | Enables ONNX inference |
| `veritrue.onnx.modelPath` | String | `classpath:models/veritrue_int8.onnx` | Location of the quantized model |
| `veritrue.onnx.fakeIndex` | Integer | `1` | Model output index corresponding to the fake class |

---

## Production Deployment

VeriTrue supports deployment using Vercel for the frontend and Render for the backend.

### Deploying the Frontend to Vercel

1. Connect the GitHub repository to Vercel.
2. Set the project root directory to `frontend`.
3. Open the Environment Variables section.
4. Add the following variable:

```env
VITE_API_BASE_URL=https://veritrue-backend.onrender.com
```

5. Deploy the application.

Vercel can automatically trigger new deployments when changes are pushed to the connected repository.

### Deploying the Backend to Render

1. Create a new Web Service on Render.
2. Connect the GitHub repository.
3. Set the root directory to `backend`.
4. Select the Java environment.
5. Configure the build command:

```bash
./mvnw clean package -DskipTests
```

6. Configure the start command:

```bash
java -Xmx200m -Xms128m -jar target/*.jar
```

The JVM heap configuration is intended to limit memory usage in environments with a 512 MB memory boundary.

**Important:** Native ONNX Runtime memory usage also contributes to total process memory. Actual deployment stability depends on the model, workload, and available resources.

---

## API Documentation

### 1. Analyze Image Authenticity

**Endpoint**

```http
POST /api/v1/analysis/upload
```

**Content-Type**

```text
multipart/form-data
```

**Request Body**

| Parameter | Type | Description |
|---|---|---|
| `file` | Image | Uploaded image file |

Supported image formats:

- PNG
- JPG
- JPEG
- WEBP

**Example JSON Response**

```json
{
  "status": "SUCCESS",
  "filename": "sample_input.jpg",
  "prediction": "Authentic",
  "confidenceScore": 0.9642,
  "probabilities": {
    "real": 96.42,
    "fake": 3.58
  },
  "engineUsed": "ONNX_INT8_NATIVE",
  "analysisTimestamp": "2026-03-30T10:15:30Z"
}
```

### 2. System Health Check

**Endpoint**

```http
GET /api/v1/analysis/health
```

**Example Response**

```json
{
  "status": "UP",
  "onnxEngineActive": true,
  "modelPath": "classpath:models/veritrue_int8.onnx"
}
```

*The endpoint paths and response structures above follow the project documentation. Verify them against the currently deployed backend configuration.*

---

## Future Enhancements

Potential improvements for future versions include:

- Video-based deepfake detection.
- Multi-modal image and video analysis.
- Advanced forensic feature extraction.
- Support for additional AI detection models.
- Improved model accuracy and evaluation.
- Detection history and downloadable reports.
- Enhanced explainability and visualization.
- Additional model optimization techniques.

---

## Live Application

**Live Demo:** [VeriTrue – Deepfake Detection System](https://deep-fake-detection-system-seven.vercel.app/)

---

## Author

**Kambampati Gopi Sai**

- GitHub: [@kgopi122](https://github.com/kgopi122)
- Project Repository: [Deep-Fake-Detection-System](https://github.com/kgopi122/Deep-Fake-Detection-System)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

<p align="center">
  <strong>VeriTrue – Building Trust in Digital Media Through AI</strong>
  <br/>
  <sub>Developed by Kambampati Gopi Sai</sub>
</p>
