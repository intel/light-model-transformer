#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <stdio.h>
#include "Model.hpp"

static void decodeImage2Buffer(const char *file, float *buffer, int height, int width, int channels) {
    // Load image
    cv::Mat img = cv::imread(file, -1);
    if (img.channels() != channels) {
        printf("Channels does not match (image channel=%d, required channels=%d)\n",
               img.channels(), channels);
        exit(-1);
    }

    // Resize if needed
    cv::Size inputSize(width, height);
    if (img.size() != inputSize) {
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, inputSize);
        img = resizedImg;
    }

    // Convert to float data
    cv::Mat sample;
    int type = ((channels == 3) ? CV_32FC3 : CV_32FC1);
    img.convertTo(sample, type);

    // Split channels (after this, it is in CHW format)
    std::vector<cv::Mat> inputChannels;
    for (int i = 0; i < channels; ++i) {
        cv::Mat channel(height, width, CV_32FC1, buffer + i * height * width);
        inputChannels.push_back(channel);
    }
    cv::split(sample, inputChannels);
}

static void usage(char **argv) {
    printf("Usage: %s weight_file image_file [loop_count]\n", argv[0]);
}

int main(int argc, char **argv) {
	const char *weight_file;
    const char *image_file;
    float *input;
    float *output;

    if (argc < 3) {
        usage(argv);
        exit(-1);
    } else {
        weight_file = argv[1];
        image_file = argv[2];
    }

    int loop = 1;
    if (argc >= 4) {
        loop = atoi(argv[3]);
        if (loop <= 0) { loop = 1; }
    }

    // Initialize the model (batch size=1)
    Model model(weight_file, 1);

    // Prepare input
    int height = model.getInputHeight();
    int width = model.getInputWidth();
    int channels = model.getInputChannels();
    input = new float[height * width * channels];
    decodeImage2Buffer(image_file, input, height, width, channels);

    // Do the inference
    printf("\nStart running ....\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int l = 0; l < loop; l++) {
        output = model.inference(input);
    }
    gettimeofday(&end, NULL);
    float avgTime = 1.0f * ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000) / loop;

    // Print the output
    printf("Last output: [%f %f %f %f]\n", output[0], output[1], output[2], output[3]);
    printf("\nAverage Time: %.2f ms\n\n", avgTime);

    // Cleanup
    delete[] input;

    return 0;
}
