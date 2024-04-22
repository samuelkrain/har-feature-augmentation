# har-feature-augmentation

Model architectures and data for feature-augmented CNNs deployed on TinyML for use in HAR applications

## Dependencies

- [TensorFlow 2 Framework](https://www.tensorflow.org/) (Inc. with Google Colab)
- [Arduino TFLite Library](https://github.com/tensorflow/tflite-micro-arduino-examples)

## File Structure

`X` is used to as a placeholder for specific architectures or model version files.

```

│   data.csv                           // Model hyperparams and experimental data
│   har_serial_com.py                  // Modified 'TestOverSerial' Script
│   LICENSE
│   README.md
│
├───arduino
│   ├───har_model_X                    // Model type (e.g. R, AB ...)
│   .   │   arduino_main.cpp            
│   .   │   har_detection_model.cpp    // File containing model weights
│   .   │   har_detection_model.h
│       │   har_model.ino              // Main file
│       │   main_functions.h            
│       │   model_settings.h           // Global parameter definitions
│       │   test_data.cpp              // One frame of test data
│       │   test_data.h
│       │
│       └───data
│               data_combined.csv      // Data sent to MCU over Serial
│               serial_test_config.json // Config for har_serial_com.py
│   
│
├───models
│   ├───X                              // ZIP files containing models used for study
│   │       X1_2_2_8.zip               // Specific model version ZIP files (1-8)
│   .       X2_4_2_16.zip
│   .       X3_8_3_32.zip
│   .       X4_16_4_64.zip
│           X5_32_6_128.zip
│           X6_64_8_256.zip
│           X7_128_11_512.zip
│           X8_256_16_1024.zip
│   
│
├───python                              // Used to create NN models
│       cnn_uci2_f.ipynb
│       cnn_uci2_rf.ipynb
│       cnn_uci_r.ipynb
│
└───uci_data                            // Data for train, val and test datasets
        all_data.csv
        all_data_quant.csv
        all_data_test.csv
        all_data_val.csv
        answers.csv
        answers_test.csv
        answers_val.csv
        features.csv
        test_features.csv
        val_features.csv

```

## Workflow

- Modify all necessary file directories in the python files, and the serial port in `har_serial_com.py`.
- Run the `.ipynb` files in Google Colab. To run the file, a Google Drive must be connected containing the files in `uci_data`.
- This will automatically generate `har_detection_model.cpp` and `test_data.cpp` files.
- Download the ZIP file and copy the `.cpp` and `.h` files to the Arduino project directory matching the model type (R,F or A).
- Connect an Arduin
- Compile using the Arduino IDE.
- Copy `har_serial_com.py` into the Scripts subfolder of the Arduino TFLite Library
- Open the command line at the directory of the installed Arduino TFLite library.
- Run the command `python .\scripts\har_serial_com.py --example har_model --verbose all` ([more info](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/src/test_over_serial))

