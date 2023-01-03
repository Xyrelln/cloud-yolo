#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;


/**
 * ImageResizeData Save the data structure of the image after image processing
 */
class ImageResizeData
{
public:
    //Add the processed image
	void setImg(cv::Mat img);
    // Get the processed imagerocessed image
	cv::Mat getImg();
    // This function returns true when the aspect ratio of the original image is greater than the aspect ratio of the processed image
	bool isW();
    // This function returns true when the aspect ratio of the original image is greater than the aspect ratio of the processed image
	bool isH();
	// Add undefined reference to `cv::imread(std::string const&, int)'
	void setWidth(int width);
    // Get the width of the image after processing
	int getWidth();
	// Add the height of the image after processing
	void setHeight(int height);
    // Get the height of the image after processing
	int getHeight();
    // Add the width of the original image
	void setW(int w);
	// Get the width of the original image
	int getW();
	// Add the height of the original image
	void setH(int h);
	// Get the height of the original imaget of the original image
	int getH();
	// Add the size of the black border added from the original image to the processed image
	void setBorder(int border);
	// Get the size of the black border added from the original image to the processed imageborder added from the original image to the processed image
	int getBorder();
private:
	// image height after processing
	int height;
	// The width of the image after processing
    int width;
	// original image width
    int w;
	// original image height
    int h;
	// From the original image to the size of the black border added to the processed image
    int border;
    //processed image
    cv::Mat img;
};

/**
* Implementation class of YoloV5
*/
class YoloV5
{
public:
   /**
     *Constructor
     *@param ptFile yoloV5 pt file path
	 *@param isCuda whether to use cuda or not by default
 	 *@param height yoloV5 image height during training
 	 *@param width yoloV5 image width during training
	 *@param confThres scoreThresh in non-maximum suppression
	 *@param iouThres iouThresh in non-maximum suppression
     */
	YoloV5(std::string ptFile, bool isCuda = false, bool isHalf = false, int height = 640, int width = 640,  float confThres = 0.25, float iouThres = 0.45);
	/**
	*prediction function
	*@param data language prediction data format (batch, rgb, height, width)
	*/
	std::vector<torch::Tensor> prediction(torch::Tensor data);
	/**
	*prediction function
	*@param filePath The image path that needs to be predicted
	*/
	std::vector<torch::Tensor> prediction(std::string filePath);
	/**
	*prediction function
	*@param img The picture to be predicted
	*/	
	std::vector<torch::Tensor> prediction(cv::Mat img);
	/**
	*prediction function
	*@param imgs The set of images that need to be predicted
	*/
	std::vector<torch::Tensor> prediction(std::vector <cv::Mat> imgs);
	/**
	*Function to change image size
	*@param img original image
	*@param height The height of the image to be processed
	*@param width The width of the image to be processed
	*@return The encapsulated image data structure after processing
	*/
	static ImageResizeData resize(cv::Mat img, int height, int width);
	/**
	*Function to change image size
	*@param img original image
	*@return The encapsulated image data structure after processing
	*/
	ImageResizeData resize(cv::Mat img);
	/**
	*Function to change image size
	*@param imgs original image collection
	*@param height The height of the image to be processed
	*@param width The width of the image to be processed
	*@return The encapsulated image data structure after processing
	 */
	static std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs, int height, int width);
	/**
	*Function to change image size
	*@param imgs original image collection
	*@return The encapsulated image data structure after processing
	*/
	std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs);
	/**
	*Draw a box in the given picture according to the output
	*@param imgs original image collection
	*@param rectangles The results processed by the prediction function
	*@param labels category labels
	*@param thickness line width
	*@return The framed picture
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, std::string> labels, int thickness = 2);
	/**
	*Draw a box in the given picture according to the output
	*@param imgs original image collection
	*@param rectangles The results processed by the prediction function
	*@param thickness line width
	*@return The framed picture
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, int thickness = 2);
	/**
	*Draw a box in the given picture according to the output
	*@param imgs original image collection
	*@param rectangles The results processed by the prediction function
	*@param colors each type corresponds to the color
	*@param labels category labels
	*@return The framed picture
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	*Draw a box in the given picture according to the output
	*@param img original image
	*@param rectangle The result processed by the prediction function
	*@param thickness line width
	*@return The framed picture
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, int thickness = 2);
	/**
	*Draw a box in the given picture according to the output
	*@param img original image
	*@param rectangle The result processed by the prediction function
	*@param labels category labels
	*@param thickness line width
	*@return The framed picture
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2);
	/**
	*Draw a box in the given picture according to the output
	*@param img original image
	*@param rectangle The result processed by the prediction function
	*@param colors each type corresponds to the color
	*@param labels category labels
	*@param thickness line width
	*@return The framed picture
	 */
	cv::Mat	drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	*Used to determine whether a prediction exists for the given data
	*@param clazz The result processed by the prediction function
	*@return true if there is a given category in the picture
	 */
	bool existencePrediction(torch::Tensor clazz);
	/**
	*Used to determine whether a prediction exists for the given data
	*@param classes handle predicted results through prediction function
	*@return returns true if a given category exists in the image collection
	 */
	bool existencePrediction(std::vector<torch::Tensor> classs);


private:
	bool isCuda;
	bool isHalf;
	// confidence threshold
	float confThres;
	// IoU threshold
	float iouThres;
	float height;
	float width;
	// color mapping
	std::map<int, cv::Scalar> mainColors;
	// model
	torch::jit::script::Module model;
	// get random ccolor
	cv::Scalar getRandScalar();
	// image channel ensure rgb
	cv::Mat img2RGB(cv::Mat img);
	// image to Tensor
	torch::Tensor img2Tensor(cv::Mat img);
	// (center_x center_y w h) to (left, top, right, bottom)
	torch::Tensor xywh2xyxy(torch::Tensor x);
	// (left, top, right, bottom) to (center_x center_y w h)
	torch::Tensor xyxy2xywh(torch::Tensor x);
	// NMS
	torch::Tensor nms(torch::Tensor bboxes, torch::Tensor scores, float thresh);
	// prediction size back to original
	std::vector<torch::Tensor> sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs);
	// NMS for predictions
	std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float confThres = 0.25, float iouThres = 0.45);
};

// 終わらない
class LoadImages {
	public:
	// variables and assets
	std::vector<std::string> files;
	std::vector<std::string> images, videos;
	unsigned int ni = 0, nv = 0, nf = 0;  // number of images, video and total files
	unsigned int count = 0;  // a flag that stores the next file index, denotes from 0

	// Constructor
	LoadImages(std::vector<std::string> paths, int img_size=640, int stride=32) {
	
		// paths contains a sequence of .jpg strings
		// now ensure paths are absolute and push them respectively into vector "files"
		for (auto path : paths) {
			std::string p = fs::absolute(p);
			if (p.std::string::find('*') != std::string::npos) { // if '*' in p
				//TODO: wildcard 
			} else {
				files.push_back(path);
			}
		}

		// catagorize files into images or videos, push into corresponding vector
		for (auto file : files) {
			if (std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), _file_extension(file)) != IMG_FORMATS.end()) {
				images.push_back(file);
			} else if (std::find(VID_FORMATS.begin(), VID_FORMATS.end(), _file_extension(file)) != VID_FORMATS.end()) {
				videos.push_back(file);
			} else {
				std::cout << file << " is not detected as an image or a video" << std::endl;
			}
		}
		
		ni = images.size();
		nv = videos.size();
		nf = ni + nv;

		// TODO: if any videos, neoVideo()
	}
	
	private:
	// Const references
	const std::vector<std::string> IMG_FORMATS{ "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm" };  // include image suffixes
	const std::vector<std::string> VID_FORMATS{ "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"};  // include video suffixes
	
	// Functions
	cv::Mat read() {		
		cv::Mat im0;
		if (count >= nf) {			// Remember to check if returned matrix is empty.						
			return im0;				// It indicates file traverse is over.	
		}
		std::string path = files[count];
		count += 1;
		im0 = cv::imread(path);
		if (im0.data == NULL) {
			std::cout << "Invalid Image:" << path << std::endl;
			std::abort();
		} 
		return im0;
	}


	void _new_video(std::string path);

	// utils
	std::string _file_extension(std::string filename) {
		std::size_t found = filename.find_last_of('.');
		return filename.substr(found);
	}

	std::string _file_name(std::string path) const {  // filename with extension
		std::size_t found = path.find_last_of("/\\");
		return path.substr(found);
	}

};