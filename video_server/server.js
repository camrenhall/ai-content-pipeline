const express = require('express');
const fileUpload = require('express-fileupload');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

const MAX_ATTEMPTS = 20;
const DELAY_MS = 6000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure express-fileupload
app.use(fileUpload({
limits: { fileSize: 100 * 1024 * 1024 }, // 100 MB limit
  abortOnLimit: true,
  useTempFiles: true,
  tempFileDir: path.join(__dirname, 'tmp'),
  createParentPath: true,
  debug: false  // Set to false in production
}));

// Create tmp directory if it doesn't exist
const tmpDir = path.join(__dirname, 'tmp');
if (!fs.existsSync(tmpDir)) {
  fs.mkdirSync(tmpDir, { recursive: true });
}

// Simple route for testing
app.get('/', (req, res) => {
  res.send('Video Caption Service is running');
});

// Process video route
app.post('/caption-video', async (req, res) => {
  try {
    // Check if file was uploaded
    if (!req.files || !req.files.video) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoFile = req.files.video;
    
    // Move the file to ensure it has a proper file extension
    const fileExt = path.extname(videoFile.name) || '.mp4';
    const fileName = `video_${Date.now()}${fileExt}`;
    const filePath = path.join(tmpDir, fileName);
    
    await videoFile.mv(filePath);
    
    console.log(`Video saved to: ${filePath}`);
    
    // Create a server URL for the file
    const videoUrl = `${process.env.PUBLIC_BASE_URL}/tmp/${fileName}`;
    
    // Process with Creatomate API
    const captionedVideoUrl = await processThroughCreatomate(videoUrl);
    
    // Return the result
    res.json({ 
      success: true, 
      captionedVideoUrl 
    });
    
    // Clean up the temporary file (after a delay to ensure it's processed)
    setTimeout(() => {
      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting temp file:', err);
        console.log(`Deleted temporary file: ${filePath}`);
      });
    }, 180000); // 3 minute delay
    
  } catch (error) {
    console.error('Error processing video:', error);
    res.status(500).json({ 
      error: 'Failed to process video', 
      details: error.message 
    });
  }
});

// Serve the temporary files
app.use('/tmp', express.static(tmpDir));

// Function to process the video through Creatomate
async function processThroughCreatomate(videoUrl) {
  try {
    // Creatomate API request
    const response = await axios.post(
      'https://api.creatomate.com/v1/renders',
      {
        "output_format": "mp4",
        "source": {
          "elements": [
            {
              "type": "video",
              "id": "source-video",
              "source": videoUrl
            },
            {
              "type": "text",
              "transcript_source": "source-video",
              "transcript_effect": "highlight",
              "transcript_maximum_length": 14,
              "y": "82%",
              "width": "81%",
              "height": "35%",
              "x_alignment": "50%",
              "y_alignment": "50%",
              "fill_color": "#ffffff",
              "stroke_color": "#000000",
              "stroke_width": "1.6 vmin",
              "font_family": "Montserrat",
              "font_weight": "700",
              "font_size": "9.29 vmin",
              "background_color": "rgba(216,216,216,0)",
              "background_x_padding": "31%",
              "background_y_padding": "17%",
              "background_border_radius": "31%"
            }
          ]
        }
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.CREATOMATE_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    // Check if render was created successfully
    if (response.data && response.data.length > 0) {
      // Initially, the render might still be processing
      const renderId = response.data[0].id;
      
      // Poll for render completion
      return await pollRenderStatus(renderId);
    } else {
      throw new Error('No render data returned from Creatomate');
    }
  } catch (error) {
    console.error('Error in Creatomate API request:', error);
    throw error;
  }
}

// Function to poll render status until complete
async function pollRenderStatus(renderId, maxAttempts = MAX_ATTEMPTS) {
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    try {
      const response = await axios.get(
        `https://api.creatomate.com/v1/renders/${renderId}`,
        {
          headers: {
            'Authorization': `Bearer ${process.env.CREATOMATE_API_KEY}`
          }
        }
      );
      
      const renderStatus = response.data.status;
      
      if (renderStatus === 'succeeded') {
        return response.data.url;
      } else if (renderStatus === 'failed') {
        throw new Error(`Render failed: ${response.data.error_message || 'Unknown error'}`);
      }
      
      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, DELAY_MS)); // 3 seconds
      attempts++;
    } catch (error) {
      console.error('Error polling render status:', error);
      throw error;
    }
  }
  
  throw new Error('Maximum polling attempts reached');
}

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Temporary files directory: ${tmpDir}`);
});