import 'package:flutter/material.dart';
import 'second_page.dart'; // Correct import

class FrontPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[300],
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(25.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  "Mini Project",
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[900],
                  ),
                ),
                SizedBox(height: 20),
                Text(
                  "On",
                  style: TextStyle(fontSize: 26, color: Colors.blue[900]),
                ),
                SizedBox(height: 20),
                Text(
                  "Handwritten Digit Recognition",
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[900],
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 40),
                Text(
                  "Under the Supervision of",
                  style: TextStyle(fontSize: 22, color: Colors.blue[900]),
                ),
                SizedBox(height: 8),
                Text(
                  "Prof. S. R. N. Reddy",
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[900],
                  ),
                ),
                SizedBox(height: 40),
                Text(
                  "Department of Computer Science and Engineering",
                  style: TextStyle(fontSize: 22, color: Colors.blue[900]),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 30),
                Image.asset("assets/logo.png", height: 140),
                SizedBox(height: 40),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const PipelineHome()),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue[900],
                    padding: EdgeInsets.symmetric(
                      vertical: 14,
                      horizontal: 40,
                    ),
                  ),
                  child: Text(
                    "GET STARTED",
                    style: TextStyle(fontSize: 20, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
