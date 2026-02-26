import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MnistPipelineApp());
}

// 👉 Set your FastAPI backend PC IP here
const String SERVER_HOST = "xx.xx.xx.xx";
const String BASE_URL = "http://$SERVER_HOST:8000";
const String WS_URL = "ws://$SERVER_HOST:8000/ws/logs";

String? singlePrediction;

class MnistPipelineApp extends StatelessWidget {
  const MnistPipelineApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MNIST Pipeline',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const PipelineHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PipelineHome extends StatefulWidget {
  const PipelineHome({super.key});

  @override
  State<PipelineHome> createState() => _PipelineHomeState();
}

class _PipelineHomeState extends State<PipelineHome> {
  final List<String> models = ["svm_pca", "svm", "knn", "knn_pca", "cnn"];
  String selectedModel = "svm_pca";
  double testSize = 0.2;
  bool loading = false;

  String uploadedImageUrl = "";
  Uint8List? selectedImageBytes;

  List<String> logs = [];
  WebSocketChannel? channel;

  String lastArtifactImage = "";
  String reportImage = "";
  Map<String, dynamic> lastMetrics = {};

  @override
  void initState() {
    super.initState();
    _connectWs();
  }

  // ----------------------------------------------------------
  // WEBSOCKET CONNECTION
  // ----------------------------------------------------------
  void _connectWs() {
    try {
      channel = WebSocketChannel.connect(Uri.parse(WS_URL));

      channel!.stream.listen((message) {
        try {
          final Map<String, dynamic> payload = json.decode(message);
          setState(() {
            logs.add("[${payload['type']}] ${payload['message']}");
          });
        } catch (e) {
          setState(() => logs.add(message.toString()));
        }
      });
    } catch (e) {
      setState(() => logs.add("WebSocket error: $e"));
    }
  }

  // ----------------------------------------------------------
  // GENERIC POST CALL
  // ----------------------------------------------------------
  Future<void> _callPost(String endpoint, {Map<String, dynamic>? body}) async {
    setState(() => loading = true);

    try {
      final url = Uri.parse("$BASE_URL$endpoint");
      http.Response res;

      if (body == null) {
        res = await http.post(url);
      } else {
        res = await http.post(url,
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(body));
      }

      logs.add("[$endpoint] ${res.statusCode} -> ${res.body}");

      if (res.statusCode == 200) {
        final decoded = jsonDecode(res.body);

        if (decoded is Map && decoded.containsKey("metrics")) {
          setState(() => lastMetrics = decoded["metrics"]);
        }

        if (decoded is Map && decoded.containsKey("report_image")) {
          _fetchReportImage();
        }
      }
    } catch (e) {
      logs.add("ERROR $endpoint: $e");
    } finally {
      setState(() => loading = false);
    }
  }

  Future<void> _callGet(String endpoint) async {
    setState(() => loading = true);
    try {
      final res = await http.get(Uri.parse("$BASE_URL$endpoint"));
      logs.add("[GET] ${res.statusCode}");

      if (res.statusCode == 200) {
        setState(() => lastMetrics = jsonDecode(res.body));
      }
    } catch (e) {
      logs.add("GET ERROR: $e");
    } finally {
      setState(() => loading = false);
    }
  }

  // ----------------------------------------------------------
  // PIPELINE ACTIONS
  // ----------------------------------------------------------
  Future<void> _loadDataset() async => await _callPost("/load_dataset");

  Future<void> _preprocess() async => await _callPost("/preprocess",
      body: {"apply_pca": true, "n_components": 50});

  Future<void> _split() async => await _callPost("/split",
      body: {"test_size": testSize, "random_state": 42, "stratify": true});

  Future<void> _train() async {
    final body = {"model_name": selectedModel};

    if (selectedModel == "cnn") {
      body["epochs"] = "5";
    } else if (selectedModel.startsWith("knn")) {
      body["knn_k"] = "5";
    }

    await _callPost("/train", body: body);
  }

  // ORIGINAL PIPELINE PREDICT
  Future<void> _predict() async {
    try {
      var req = http.MultipartRequest("POST", Uri.parse("$BASE_URL/predict"));
      req.fields["model_name"] = selectedModel;

      var res = await req.send();
      var body = await res.stream.bytesToString();

      setState(() => logs.add("Predict response: $body"));
    } catch (e) {
      setState(() => logs.add("Predict error: $e"));
    }
  }

  // ----------------------------------------------------------
  // PREDICT SINGLE DIGIT WITH IMAGE UPLOAD
  // ----------------------------------------------------------
  Future<void> predictSingleDigit() async {
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: ImageSource.gallery);

      if (picked == null) {
        logs.add("No image selected.");
        return;
      }

      final bytes = await picked.readAsBytes();

      setState(() {
        selectedImageBytes = bytes;
        uploadedImageUrl = "";
      });

      var request = http.MultipartRequest(
        "POST",
        Uri.parse("$BASE_URL/predict_single?model_name=$selectedModel"),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          "file",
          bytes,
          filename: "digit.png",
        ),
      );

      final response = await request.send();
      final raw = await response.stream.bytesToString();

      logs.add("RAW RESPONSE: $raw");

      final data = jsonDecode(raw);

      if (data["status"] == "ok") {
        setState(() {
          uploadedImageUrl = "$BASE_URL${data['image_url']}";
          singlePrediction = data["prediction"].toString();
        });
      } else {
        logs.add("Prediction error: ${data['detail']}");
      }
    } catch (e) {
      logs.add("EXCEPTION: $e");
    }
  }

  // ----------------------------------------------------------
  // FETCH IMAGES
  // ----------------------------------------------------------
  Future<void> _fetchConfusionImage() async {
    await _fetchImage("${selectedModel}_confusion.png", target: "confusion");
  }

  Future<void> _fetchReportImage() async {
    await _fetchImage("${selectedModel}_classification_report.png",
        target: "report");
  }

  Future<void> _fetchImage(String filename, {required String target}) async {
    try {
      final Uri url = Uri.parse("$BASE_URL/artifacts/$filename");

      final res = await http.get(url);

      if (res.statusCode == 200) {
        setState(() {
          if (target == "confusion") {
            lastArtifactImage = base64Encode(res.bodyBytes);
          } else {
            reportImage = base64Encode(res.bodyBytes);
          }
        });
      } else {
        logs.add("Image fetch failed: ${res.statusCode}");
      }
    } catch (e) {
      logs.add("IMAGE ERROR: $e");
    }
  }

  // ----------------------------------------------------------
  Widget _img(String base64) {
    if (base64.isEmpty) {
      return Container(
        height: 200,
        color: Colors.grey.shade300,
        child: const Center(child: Text("No image")),
      );
    }
    final bytes = base64Decode(base64);
    return Image.memory(Uint8List.fromList(bytes),
        height: 220, fit: BoxFit.contain);
  }

  // ----------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[200],
      appBar: AppBar(
        title: const Text("MNIST Model Training Pipeline",
            style: TextStyle(color: Colors.blue)),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 20),

            _btn("1. Load Dataset", _loadDataset),
            _btn("2. Preprocessing (PCA)", _preprocess),
            _btn("3. Train/Test Split", _split),
            _btn("4. Train Model", _train),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                DropdownButton(
                  value: selectedModel,
                  items: models
                      .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                      .toList(),
                  onChanged: (v) => setState(() => selectedModel = v!),
                ),
              ],
            ),

            _btn("5. Predict (Pipeline)", _predict),
            _btn("6. Results (Metrics)", () async {
              await _callGet("/results?model_name=$selectedModel");
            }),
            _btn("7. Show Plots", () async {
              await _fetchConfusionImage();
              await _fetchReportImage();
            }),

            // _btn("8. Predict Single Digit (Upload Image)", predictSingleDigit),

            // ------------------------------------------------------
            // SHOW SELECTED IMAGE (before sending to backend)
            // ------------------------------------------------------
            if (selectedImageBytes != null) ...[
              const SizedBox(height: 15),
              const Text("Selected Image",
                  style: TextStyle(color: Colors.blue, fontSize: 18)),
              const SizedBox(height: 10),
              Image.memory(
                selectedImageBytes!,
                height: 200,
                fit: BoxFit.contain,
              ),
            ],

            // ------------------------------------------------------
            // SHOW UPLOADED IMAGE
            // ------------------------------------------------------
            if (uploadedImageUrl.isNotEmpty) ...[
              const SizedBox(height: 15),
              const Text("Uploaded Image",
                  style: TextStyle(color: Colors.blue, fontSize: 18)),
              const SizedBox(height: 10),
              Image.network(
                uploadedImageUrl,
                height: 200,
                fit: BoxFit.contain,
                errorBuilder: (c, e, s) => Text("Image load error"),
              ),
            ],

            // ------------------------------------------------------
            // SHOW PREDICTION
            // ------------------------------------------------------
            if (singlePrediction != null) ...[
              const SizedBox(height: 15),
              Text(
                "Predicted Digit: $singlePrediction",
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.red),
              ),
            ],

            const SizedBox(height: 25),

            // ------------------------------------------------------
            // LOGS
            // ------------------------------------------------------
            Container(
              height: 200,
              width: double.infinity,
              margin: const EdgeInsets.all(10),
              padding: const EdgeInsets.all(8),
              color: Colors.black,
              child: SingleChildScrollView(
                child: Text(
                  logs.join("\n"),
                  style: const TextStyle(color: Colors.greenAccent),
                ),
              ),
            ),

            const SizedBox(height: 15),

            const Text("Confusion Matrix",
                style: TextStyle(color: Colors.blue)),
            _img(lastArtifactImage),

            const SizedBox(height: 20),

            const Text("Classification Report",
                style: TextStyle(color: Colors.blue)),
            _img(reportImage),

            const SizedBox(height: 20),

            const Text("Metrics", style: TextStyle(color: Colors.blue)),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                jsonEncode(lastMetrics),
                style: const TextStyle(color: Colors.blue),
              ),
            ),

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _btn(String text, VoidCallback onPress) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: SizedBox(
        width: 240,
        height: 50,
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(backgroundColor: Colors.blue[800]),
          onPressed: loading ? null : onPress,
          child: Text(text, style: const TextStyle(color: Colors.white)),
        ),
      ),
    );
  }
}
