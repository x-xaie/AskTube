# AskTube

**AskTube** is an application that harnesses the power of OpenAI's GPT and Streamlit's user-friendly interface to revolutionize the way users interact with YouTube videos. With AskTube, users can effortlessly query YouTube videos in a convenient question-and-answer format, streamlining the process of extracting valuable information from video content.
 
To begin using AskTube, users simply access the provided URL and enter the YouTube video URL of interest. OpenAI's advanced algorithms analyze the input, allowing users to ask questions or queries about the video content. The application then displays precise and accurate answers generated by OpenAI.

In addition to the impressive capabilities of OpenAI's GPT, AskTube incorporates pyttsx3, a text-to-speech library, to enhance the user experience. The answers provided by OpenAI are visually displayed and read aloud, creating an immersive and inclusive browsing experience. However, further improvements are underway to enhance the functionality and user interface of AskTube, ensuring an even more seamless and engaging experience for users as they explore YouTube videos.


## Dependencies
* pip install --upgrade pip
* pip install youtube-transcript-api pip install openai
* pip install chromadb
* pip install langchain
* pip install streamlit
* pip install tiktoken
* pip install pyttsx3

## Installation
* Install the required dependencies. Run the following command in your terminal or command prompt
* Set up OpenAI API credentials
* Obtain an API key from OpenAI
* Set up OpenAI API credentials
* Obtain an API key from OpenAI

## Usage
* Navigate to the project directory in your terminal or command prompt.
* Run the following command to start the application:
*  *streamlit run app.py
* Access the application in your web browser
* Enter the URL of the YouTube video you want to query and press Enter.
* Ask your question or query in the provided input field
* The application will display the answer generated by OpenAI. Additionally, the answer will be read aloud using pyttsx3.
* Explore other YouTube videos by entering their URLs and repeating the process.

## Future Improvements
While the current version of AskTube provides basic functionality, there are several areas for future improvements:
* Enhance the user interface for a more visually appealing and intuitive experience.
* Implement advanced language processing techniques to improve the accuracy and relevance of answers.
* Optimize the application for faster performance and better scalability.
* Incorporate additional features such as video recommendations or topic clustering based on user queries.
