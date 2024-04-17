/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>
#include <stdio.h>
#include <cmath>

#include "main_functions.h"
#include "har_detection_model.h"
#include "data_handler.h"
#include "model_settings.h"
#include "test_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "test_over_serial/test_over_serial.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define CALCULATE_FEATS 1


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
uint8_t* model_input_buffer = nullptr;
uint8_t* model_output_buffer = nullptr;

int inference_count = 0;
bool useSampleData = 1;
bool receivedNewSample = 0;
unsigned long start_time = 0;
unsigned long start_inf_time = 0;
unsigned long finish_time = 0;
unsigned long int delta_inf_time = 0;
unsigned long delta_time = 0;

unsigned long pre_mean,post_mean,t_mean;
unsigned long pre_std,post_std,t_std;
unsigned long pre_mae,post_mae,t_mae;
unsigned long pre_rms,post_rms,t_rms;
unsigned long pre_hist,post_hist,t_hist;
unsigned long pre_featureprocess,post_featureprocess,t_featureprocess;




// For feature generation
double tmp_sum[3];
double feat_mean[3];
double feat_std[3];
double feat_mae[3];
double feat_rms;
float tmp_min[3],tmp_max[3],range[3],bin_width[3];
int feat_hist[3][10];


// Takes input1, input2, label over serial
constexpr int kInputBufferSize = (kInputElementCount+kFeatureElementCount+1);
float g_input_buffer[kInputBufferSize];
int read_buffer_section = 0;
int g_test_sample_index = 0;

constexpr int kTensorArenaSize = 100*1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

using namespace test_over_serial;

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Add required operations
  static tflite::MicroMutableOpResolver<12> micro_op_resolver;
  if (micro_op_resolver.AddExpandDims() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddShape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddStridedSlice() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddConcatenation() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSqueeze() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddAdd() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  model_input_buffer = interpreter->input(0)->data.uint8;
  model_output_buffer = interpreter->output(0)->data.uint8;

  // Get information about the memory area to use for the model's input.
  if ((interpreter->input(0)->dims->size != 2) || 
      (interpreter->input(0)->type != kTfLiteUInt8)) {
    MicroPrintf("Bad input tensor parameters in model\n");
    MicroPrintf("Input dimension size: %d\n", interpreter->input(0)->dims->size);
    MicroPrintf("Input type: %d\n", interpreter->input(0)->type);
    return;
  }

  MicroPrintf("Initialisation complete!");
}

// The name of this function is important for Arduino compatibility.
void loop() {
  

  // Serial processing
  TestOverSerial& test = TestOverSerial::Instance(kRAW_FLOAT);

  if (!test.IsTestMode()) {
    // check serial port for test mode command
    test.ProcessInput(nullptr);
    if (!test.IsTestMode()) {
      useSampleData = 1;
    } else {
      useSampleData = 0;
    }
  }
  else if (test.IsTestMode()) {
    // Get data over serial
    useSampleData = 0;
    InputHandler handler = [](const InputBuffer* const input) {
      for (size_t i = 0; i < input->length; i++) {
      const size_t index = (g_test_sample_index + i) % kInputBufferSize;
      g_input_buffer[index] = input->data.float32[i];
      if (index % kInputSampleSize == kInputSampleSize-1) {
        receivedNewSample = 1;
      }
    }

    g_test_sample_index += input->length;
    return true;
    };

    test.ProcessInput(&handler);
  } else {
    MicroPrintf("Error when checking for test mode!");
  }

  // Run inference
  if (useSampleData || receivedNewSample) {
    int expected_result = -1;
    start_time = millis();
    pre_featureprocess = millis();

    // With serial data
    if (receivedNewSample) {
      float sc1 = interpreter->input(0)->params.scale;
      int32_t z_point1 = interpreter->input(0)->params.zero_point;
  
      if (CALCULATE_FEATS) {
        // Mean
        MicroPrintf("---features[0:2]: mean---\n");
        pre_mean = millis();
        tmp_sum[0] = 0; tmp_sum[1] = 0; tmp_sum[2] = 0;

        for (int i = 0;i<128;i++) {
          tmp_sum[0] += g_input_buffer[i] ;
          tmp_sum[1] += g_input_buffer[i+128];
          tmp_sum[2] += g_input_buffer[i+256];
        }


        feat_mean[0] = tmp_sum[0] / 128;
        feat_mean[1] = tmp_sum[1] / 128;
        feat_mean[2] = tmp_sum[2] / 128;

        model_input_buffer[0] = feat_mean[0] / sc1 + z_point1;
        model_input_buffer[1] = feat_mean[1] / sc1 + z_point1;
        model_input_buffer[2] = feat_mean[2] / sc1 + z_point1;
        post_mean = millis();
        t_mean = post_mean - pre_mean;
        MicroPrintf("t_mean: %d - %d = %d", post_mean,pre_mean,t_mean);

        MicroPrintf("float in 1: %f, 2: %f, 3: %f\n",g_input_buffer[0],g_input_buffer[1],g_input_buffer[2]);
        MicroPrintf("raw means: %f, 2: %f, 3: %f\n",feat_mean[0],feat_mean[1],feat_mean[2]);
        MicroPrintf("Quant means: %d, 2: %d, 3: %d\n",model_input_buffer[0],model_input_buffer[1],model_input_buffer[2]);
        
        // Std
        MicroPrintf("---features[3:5]: std---\n");
        pre_std = millis();
        tmp_sum[0] = 0; tmp_sum[1] = 0; tmp_sum[2] = 0;

        for (int i = 0;i<128;i++) {
          tmp_sum[0] += pow((g_input_buffer[i]-feat_mean[0]),2.0);
          tmp_sum[1] += pow((g_input_buffer[i+1*128]-feat_mean[1]),2.0);
          tmp_sum[2] += pow((g_input_buffer[i+2*128]-feat_mean[2]),2.0);
        }
        feat_std[0] = sqrt(tmp_sum[0]/128);
        feat_std[1] = sqrt(tmp_sum[1]/128);
        feat_std[2] = sqrt(tmp_sum[2]/128);

        model_input_buffer[3] = feat_std[0] / sc1 + z_point1;
        model_input_buffer[4] = feat_std[1] / sc1 + z_point1;
        model_input_buffer[5] = feat_std[2] / sc1 + z_point1;

        MicroPrintf("Stdevs = %f, %f, %f",feat_std[0],feat_std[1],feat_std[2]);
        MicroPrintf("Quant_std 1 = %d, %d, %d",model_input_buffer[3],model_input_buffer[4],model_input_buffer[5]);
        post_std = millis();
        t_std = post_std-pre_std;
        MicroPrintf("t_std: %d - %d = %d\n",post_std,pre_std,t_std);

        // mae
        MicroPrintf("---features[6:8]: mae---\n");
        pre_mae = millis();
        tmp_sum[0] = 0; tmp_sum[1] = 0; tmp_sum[2] = 0;

        for (int i = 0;i<128;i++) {
          // mean(abs(x - mean(x)))
          tmp_sum[0] += abs(g_input_buffer[i]-feat_mean[0]);
          tmp_sum[1] += abs(g_input_buffer[i+1*128]-feat_mean[1]);
          tmp_sum[2] += abs(g_input_buffer[i+2*128]-feat_mean[2]);
        }
        feat_mae[0] = tmp_sum[0]/128;
        feat_mae[1] = tmp_sum[1]/128;
        feat_mae[2] = tmp_sum[2]/128;

        model_input_buffer[6] = feat_mae[0] / sc1 + z_point1;
        model_input_buffer[7] = feat_mae[1] / sc1 + z_point1;
        model_input_buffer[8] = feat_mae[2] / sc1 + z_point1;

        MicroPrintf("mae = %f, %f, %f",feat_mae[0],feat_mae[1],feat_mae[2]);
        MicroPrintf("Quant_mae 1 = %d, %d, %d",model_input_buffer[6],model_input_buffer[7],model_input_buffer[8]);
        post_mae = millis();
        t_mae = post_mae-pre_mae;
        MicroPrintf("t_mae: %d - %d = %d\n",post_mae,pre_mae,t_mae);

        // rms and bin_width for histogram
        MicroPrintf("---features[9]: rms---\n");
        pre_rms = millis();

        tmp_sum[0] = 0;
        for (int i=0;i<3;i++) {
          tmp_max[i] = g_input_buffer[128*i];
          tmp_min[i] = g_input_buffer[128*i];
          for (int j=0;j<10;j++) {
            feat_hist[i][j] = 0;
          }
        }

        for (int i = 0;i<128;i++) {
          // sqrt(x.^2 + y.^2 + z.^2)
          tmp_sum[0] += sqrt(pow(g_input_buffer[i],2) + pow(g_input_buffer[i+1*128],2)+pow(g_input_buffer[i+2*128],2));

          for (int j=0;j<3;j++) {
            if (g_input_buffer[i+j*128] > tmp_max[j]) {
              tmp_max[j] = g_input_buffer[i+j*128];
            } else if (g_input_buffer[i+j*128] < tmp_min[j]) {
              tmp_min[j] = g_input_buffer[i+j*128];
            }
          }
        }
        feat_rms = tmp_sum[0]/128;
        
        for (int i=0;i<3;i++) {
          range[i] = tmp_max[i] - tmp_min[i];
          // MicroPrintf("tmp_max=%f,tmp_min=%f,range=%f",tmp_max[i],tmp_min[i],range[i]);
          bin_width[i] = range[i]/10;
        }

        model_input_buffer[9] = feat_rms / sc1 + z_point1;

        MicroPrintf("rms = %f",feat_rms);
        MicroPrintf("Quant_rms = %d",model_input_buffer[9]);
        post_rms = millis();
        t_rms = post_rms-pre_rms;
        MicroPrintf("t_rms: %d - %d = %d\n",post_rms,pre_rms,t_rms);

        // histogram
        MicroPrintf("---features[10:29]: hist---\n");
        pre_hist = millis();
        for (int j = 0;j<3;j++) {
          for (int i = 0;i<128;i++) {
            for (int n_bin=0;n_bin<10;n_bin++) {
              if ((g_input_buffer[i+j*128] >= (tmp_min[j]+n_bin*bin_width[j])) && 
                  (g_input_buffer[i+j*128] < tmp_min[j]+(n_bin+1)*bin_width[j])) {
                feat_hist[j][n_bin]++;
                break;
              }
              if (g_input_buffer[i+j*128] == tmp_max[j]) {
                feat_hist[j][9]++;
                break;
              }
            }
          }
        }
        
        for (int i=0;i<10;i++){
          model_input_buffer[10+i] = feat_hist[0][i] / sc1 + z_point1;
          model_input_buffer[20+i] = feat_hist[1][i] / sc1 + z_point1;
          model_input_buffer[30+i] = feat_hist[2][i] / sc1 + z_point1;
        }

        MicroPrintf("Bin width: %f, %f, %f",bin_width[0],bin_width[1],bin_width[2]);
        // MicroPrintf("max hist: %d",max(feat_hist[0]));
        MicroPrintf("hist = %d, %d, %d",feat_hist[0][0],feat_hist[0][1],feat_hist[0][2]);
        MicroPrintf("Quant_hist = %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",model_input_buffer[10],model_input_buffer[11],model_input_buffer[12],model_input_buffer[13]
                    ,model_input_buffer[14],model_input_buffer[15],model_input_buffer[16],model_input_buffer[17],model_input_buffer[18],model_input_buffer[19]);
        post_hist = millis();
        t_hist = post_hist-pre_hist;
        MicroPrintf("t_hist: %d - %d = %d\n",post_hist,pre_hist,t_hist);
        post_featureprocess = millis();
        t_featureprocess = post_featureprocess-pre_featureprocess;
        MicroPrintf("t_featureprocess: %d - %d = %d\n",pre_featureprocess,post_featureprocess,t_featureprocess);
      } else {
        for (int i=0;i<kFeatureElementCount;i++) {
          model_input_buffer[i] =  g_input_buffer[i] / sc1 + z_point1;
        }
      }
      expected_result = std::round(g_input_buffer[kInputElementCount+kFeatureElementCount]);

    // With test data
    } else if (useSampleData) {
      for (int i=0;i<kFeatureElementCount;i++) {
        model_input_buffer[i] = test_input1[i]; /// <Deleted definition?>
      }  
      expected_result = 4;   
    }    

    start_inf_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed on input");
      return;
    }

    // Display output and find corresponding label
    MicroPrintf("Likelihood scores:");
    int prediction = 0;
    for (int i=0;i<6;i++) {
      MicroPrintf("  Output %d: %d",i,model_output_buffer[i]);
      if (model_output_buffer[i] > model_output_buffer[prediction]) {
        prediction = i;
      }
    }

    finish_time = millis();
    delta_time = finish_time-start_time;
    delta_inf_time = finish_time-start_inf_time;
    MicroPrintf("Time for inference: %dms", delta_inf_time);
    MicroPrintf("Detected %d in %dms (expected %d)",prediction,delta_time,expected_result);
    
    receivedNewSample = 0;

  }

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
}


