#include "../include/load_MNIST.hpp"

using namespace cv;

struct MNISTImageFileHeader
{
    unsigned char MagicNumber[4];
    unsigned char NumberOfImages[4];
    unsigned char NumberOfRows[4];
    unsigned char NumberOfColums[4];
};

struct MNISTLabelFileHeader
{
    unsigned char MagicNumber[4];
    unsigned char NumberOfLabels[4];
};

const int MAGICNUMBEROFIMAGE = 2051;
const int MAGICNUMBEROFLABEL = 2049;

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray)
{
    if (LengthOfArray < 0)
    {
        return -1;
    }
    int result = static_cast<signed int>(array[0]);
    for (int i = 1; i < LengthOfArray; i++)
    {
        result = (result << 8) + array[i];
    }
    return result;
}

/**
 * @brief IsImageDataFile  Check the input MagicNumber is equal to
 *                         MAGICNUMBEROFIMAGE
 * @param MagicNumber      The array of the magicnumber to be checked
 * @param LengthOfArray    The length of the array
 * @return true, if the magcinumber is mathed;
 *         false, otherwise.
 */
bool IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray)
{
    int MagicNumberOfImage = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
    if (MagicNumberOfImage == MAGICNUMBEROFIMAGE)
    {
        return true;
    }
 
    return false;
}

/**
 * @brief IsImageDataFile  Check the input MagicNumber is equal to
 *                         MAGICNUMBEROFLABEL
 * @param MagicNumber      The array of the magicnumber to be checked
 * @param LengthOfArray    The length of the array
 * @return true, if the magcinumber is mathed;
 *         false, otherwise.
 */
bool IsLabelDataFile(unsigned char *MagicNumber, int LengthOfArray)
{
    int MagicNumberOfLabel = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
    if (MagicNumberOfLabel == MAGICNUMBEROFLABEL)
    {
        return true;
    }

    return false;
}

/**
 * @brief ReadData  Read the data in a opened file
 * @param DataFile  The file which the data is read from.
 * @param NumberOfData  The number of the data
 * @param DataSizeInBytes  The size fo the every data
 * @return The Mat which rows is a data,
 *         Return a empty Mat if the file is not opened or the some flag was
 *         seted when reading the data.
 */

cv::Mat ReadData(std::fstream& DataFile, int NumberOfData, int DataSizeInBytes)
{
    cv::Mat DataMat;
    // read the data if the file is opened.
    if (DataFile.is_open())
    {
        int AllDataSizeInBytes = DataSizeInBytes * NumberOfData;
        unsigned char* TmpData = new unsigned char[AllDataSizeInBytes];
        DataFile.read((char *)TmpData, AllDataSizeInBytes);
        DataMat = cv::Mat(NumberOfData, DataSizeInBytes, CV_8UC1, TmpData).clone();
        delete [] TmpData;
        DataFile.close();
    }
    return DataMat;
}

/**
 * @brief ReadImageData  Read the Image data from the MNIST file.
 * @param ImageDataFile  The file which contains the Images.
 * @param NumberOfImages The number of the images.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat is the file is closed or the data is not matching
 *                the number.
 */
cv::Mat ReadImageData(std::fstream& ImageDataFile, int NumberOfImages)
{
    int ImageSizeInBytes = 28 * 28;
    return ReadData(ImageDataFile, NumberOfImages, ImageSizeInBytes);
}

/**
 * @brief ReadLabelData Read the label data from the MNIST file.
 * @param LabelDataFile The file contained the labels.
 * @param NumberOfLabel The number of the labels.
 * @return The mat contains the labels and each row of the mat is a label.
 *         Return empty mat is the file is closed or the data is not matching
 *         the number.
 */
cv::Mat ReadLabelData(std::fstream& LabelDataFile, int NumberOfLabel)
{
    int LabelSizeInBytes = 1;
    return ReadData(LabelDataFile, NumberOfLabel, LabelSizeInBytes);
}

/**
 * @brief ReadImages Read the Training images.
 * @param FileName  The name of the file.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat if the file is closed or the data is not matched.
 */
cv::Mat ReadImages(std::string& FileName)
{
    std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!File.is_open()) return cv::Mat();

    MNISTImageFileHeader FileHeader;
    File.read((char *)(&FileHeader), sizeof(FileHeader));

    if (!IsImageDataFile(FileHeader.MagicNumber, 4)) return cv::Mat();

    int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfImages, 4);

    return ReadImageData(File, NumberOfImage);
}

/**
 * @brief ReadLabels  Read the label from the MNIST file.
 * @param FileName  The name of the file.
 * @return The mat contains the image and each row of the mat is a image.
 *         Return empty mat is the file is closed or the data is not matched.
 */
cv::Mat ReadLabels(std::string& FileName)
{
    std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!File.is_open()) return cv::Mat();

    MNISTLabelFileHeader FileHeader;
    File.read((char *)(&FileHeader), sizeof(FileHeader));

    if (!IsLabelDataFile(FileHeader.MagicNumber, 4)) return cv::Mat();

    int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfLabels, 4);

    return ReadLabelData(File, NumberOfImage);
}

void show100Images(cv::Mat images)
{
    int blank=4;//图片之间间隔
    int width=(28+blank),height=(28+blank);
    Mat imgPlay(height*10,height*10,CV_8U,Scalar(255));
    for(int i=0;i<100;i++)
    {
        int y=i/10,x=i%10;
        Mat temp=imgPlay(Rect(x*width,y*height,width,height));
        Mat mi,m;
        m=Mat(28,28,CV_8U,images.row(i).data);
        copyMakeBorder(m,mi,0,blank,0,blank,BORDER_CONSTANT,Scalar(255));
        mi.copyTo(temp);
    }
    imshow("image demo",imgPlay);
    waitKey();
}