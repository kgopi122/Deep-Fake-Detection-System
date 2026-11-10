VeriTrue: Multi-Modal AI Deepfake & Image Authenticity Detection SystemVeriTrue is an end-to-end multi-modal image authenticity and deepfake detection platform designed to detect synthetic, manipulated, and AI-generated content (AIGC). Built with a decoupled architecture featuring a React frontend and a robust Spring Boot backend, VeriTrue executes neural network inference directly in-process using an optimized, INT8-quantized ONNX Runtime engine.By eliminating the overhead of external microservice HTTP calls and optimizing the AI model footprint down to ~20MB, VeriTrue achieves low-latency detection while running seamlessly within tight memory constraints (such as cloud platform free tiers with $\le$ 512 MB RAM).🌟 Key FeaturesIn-Process AI Inference: Neural network inference is executed directly inside the Java Virtual Machine via native C++ bindings provided by ONNX Runtime Java, eliminating external microservice dependencies.INT8 Quantization Pipeline: Custom Python quantization workflow powered by onnxruntime.quantization compresses standard PyTorch weights (.pt/.pth) into a lightweight INT8 ONNX graph (veritrue_int8.onnx), shrinking model disk footprint by ~75% and dramatically reducing memory consumption during execution.Resilient Dual-Engine Architecture: Features an automated fallback mechanism: if the ONNX binary is unavailable or inference fails, the system gracefully degrades to a heuristic forensic analysis engine without throwing 500 server errors.Memory-Optimized Deployment: Specifically tuned JVM parameters (-Xmx200m -Xms128m) ensure full platform stability within 512 MB RAM cloud instances (e.g., Render Free Tier).Modern React Interface: Ultra-responsive web frontend featuring real-time image uploads, dynamic probability breakdowns, forensic feedback metrics, and environment-aware API resolution.🏗️ System Architecture                                  +---------------------------------------+
                                  |            User Browser               |
                                  +---------------------------------------+
                                                      |
                                         HTTP POST /api/v1/analysis
                                                      v
                                  +---------------------------------------+
                                  |         Vercel (React App)            |
                                  +---------------------------------------+
                                                      |
                                              REST API Requests
                                                      v
  +--------------------------------------------------------------------------------------------------+
  | Render (Spring Boot Application Container - Max 512MB RAM)                                      |
  |                                                                                                  |
  |   +--------------------------+         +-----------------------------------------------------+   |
  |   | REST Controller          | ------> |  Analysis Service Engine                            |   |
  |   +--------------------------+         +-----------------------------------------------------+   |
  |                                                                  |                               |
  |                                          +-----------------------+-----------------------+       |
  |                                          | (Primary Path)                                |       |
  |                                          v                                               v       |
  |                       +-------------------------------------+         +----------------------+   |
  |                       |  ONNX Runtime Java Engine           |         | Heuristic Forensics  |   |
  |                       |  (veritrue_int8.onnx)               |         | Fallback Engine      |   |
  |                       |  - RGB Rescale (224x224)            |         +----------------------+   |
  |                       |  - ImageNet Normalization           |                                  |
  |                       |  - In-Process C++ Inference         |                                  |
  |                       +-------------------------------------+                                  |
  |                                          |                                                       |
  +------------------------------------------|-------------------------------------------------------+
                                             v
                                  +--------------------+
                                  | Detection Result   |
                                  | { Real %, Fake % } |
                                  +--------------------+
💻 Tech StackFrontendFramework: React.js (Vite / Create React App)HTTP Client: Axios / Native Fetch APIDeployment Platform: VercelBackendLanguage & Runtime: Java 17+ (JDK 17)Framework: Spring Boot 3.xBuild System: Apache Maven (./mvnw)AI Runtime Binding: com.microsoft.onnxruntime:onnxruntime (v1.17.0+)Deployment Platform: Render (Web Service)AI / Machine Learning PipelineFramework: PyTorch (Model Definition & Training)Interoperability & Quantization: ONNX (Open Neural Network Exchange), ONNXRuntime (onnxruntime.quantization)Model Format: INT8 Dynamic Quantized Graph (QUInt8)Target Tensor Dimensions: [1, 3, 224, 224] (BCHW format)📁 Repository StructurePlaintextveritrue/
├── backend/                             # Spring Boot Application
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/veritrue/      # Controllers, Services, ONNX Loaders
│   │   │   └── resources/
│   │   │       ├── application.properties
│   │   │       └── models/
│   │   │           └── veritrue_int8.onnx  # Quantized ONNX Model (Generated)
│   ├── pom.xml                          # Maven Dependencies (ONNX Runtime Java)
│   └── mvnw                             # Maven Wrapper
├── frontend/                            # React Frontend Application
│   ├── src/
│   │   ├── components/                  # UI Components (Dashboard, Upload Zone)
│   │   ├── services/api.js              # Centralized Axios API Config
│   │   └── App.jsx                      # Main React Application Entrypoint
│   ├── package.json
│   └── .env.example
├── models/                              # Local PyTorch Model Directory
│   └── model.pt                         # Trained PyTorch Model Weights (.pt / .pth)
├── scripts/
│   └── export_quantize.py               # PyTorch to INT8 ONNX Conversion Script
└── README.md                            # Comprehensive System Documentation
⚙️ Model Quantization & Export WorkflowThe project converts raw PyTorch weights into a compressed INT8 ONNX binary file using scripts/export_quantize.py. This script includes fail-fast validation, automatic checkpoint discovery, dynamic quantization, and post-export logit verification.1. Install Python DependenciesBashpip install torch onnx onnxruntime pillow numpy
2. Export & Quantize the PyTorch ModelTo convert your trained PyTorch checkpoint (models/model.pt) to backend/src/main/resources/models/veritrue_int8.onnx, run:Bash# Explicit path execution
python scripts/export_quantize.py --pt-path models/model.pt

# Or allow auto-discovery from default search paths
python scripts/export_quantize.py
3. Verification Log OutputUpon successful export, the script automatically validates the model execution:Plaintext[INFO] Exporting PyTorch model to FP32 ONNX format...
[INFO] Quantizing ONNX model using QUInt8 dynamic weight quantization...
[INFO] Successfully generated INT8 ONNX model at: backend/src/main/resources/models/veritrue_int8.onnx
[VERIFICATION] Running sample inference pass on exported model...
[VERIFICATION] Raw Logits: [-1.4021, 2.1584]
[VERIFICATION] Softmax Probabilities -> Real: 2.76%, Fake: 97.24%
[VERIFICATION] Model Sanity Check PASSED!
🚀 Local Development SetupPrerequisitesJava Development Kit (JDK): Version 17 or higherNode.js: Version 18.x or higherPython: Version 3.10+ (for model quantization)Git1. Backend Setup (Spring Boot)Navigate to the backend directory:Bashcd backend
Verify that the quantized model veritrue_int8.onnx exists in src/main/resources/models/. If missing, follow the quantization steps above.Configure src/main/resources/application.properties:Propertiesserver.port=8000
veritrue.onnx.enabled=true
veritrue.onnx.modelPath=classpath:models/veritrue_int8.onnx
veritrue.onnx.fakeIndex=1
Build and run the Spring Boot application:Bash./mvnw clean compile
./mvnw spring-boot:run
The backend service will start on http://localhost:8000.2. Frontend Setup (React)Open a new terminal and navigate to the frontend directory:Bashcd frontend
Install npm packages:Bashnpm install
Create a .env file in the frontend/ directory (optional for local dev):Code snippetVITE_API_BASE_URL=http://localhost:8000
Start the Vite development server:Bashnpm run dev
Access the frontend user interface at http://localhost:5173.📋 Environment ConfigurationsFrontend (frontend/)Variable NameEnvironmentDescriptionExample / Recommended ValueVITE_API_BASE_URLProduction (Vercel)Live URL of deployed Spring Boot backend. Do not append paths or trailing slashes.[https://veritrue-backend.onrender.com](https://veritrue-backend.onrender.com)VITE_API_BASE_URLLocal DevAddress of local Spring Boot instance.http://localhost:8000Backend (backend/src/main/resources/application.properties)Property KeyTypeDefault ValueDescriptionveritrue.onnx.enabledBooleantrueEnables native ONNX Runtime Java inference engine.veritrue.onnx.modelPathStringclasspath:models/veritrue_int8.onnxClasspath location of the quantized ONNX binary.veritrue.onnx.fakeIndexInteger1Model output tensor index corresponding to "Fake" class.☁️ Production Deployment GuideDeploying Frontend to VercelConnect your GitHub repository to Vercel.Set the Root Directory to frontend.In Environment Variables, configure:Name: VITE_API_BASE_URLValue: [https://veritrue-backend.onrender.com](https://veritrue-backend.onrender.com)Click Deploy. Vercel will automatically trigger builds on git pushes modifying the frontend/ directory.Deploying Backend to Render (Free Tier)Create a new Web Service on Render and link your GitHub repository.Set Root Directory to backend.Select Environment: Java.Configure Build and Start commands:Build Command: ./mvnw clean package -DskipTestsStart Command (CRITICAL FOR MEMORY LIMITS):Bashjava -Xmx200m -Xms128m -jar target/*.jar
Note: Capping JVM heap at 200 MB ensures total container usage (Java + native C++ ONNX memory buffers) remains safely under Render's 512 MB memory boundary.Save and deploy.📡 API Endpoint Reference1. Analyze Image AuthenticityEndpoint: POST /api/v1/analysis/upload (or configured base route)Content-Type: multipart/form-dataRequest Body:file: Image binary (.png, .jpg, .jpeg, .webp)Sample JSON ResponseJSON{
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
2. System Health CheckEndpoint: GET /api/v1/analysis/healthResponse:JSON{
  "status": "UP",
  "onnxEngineActive": true,
  "modelPath": "classpath:models/veritrue_int8.onnx"
}
👨‍💻 AuthorKambampati Gopi SaiGitHub: @kgopi122Project Repository: https://github.com/kgopi122/veritrue📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
