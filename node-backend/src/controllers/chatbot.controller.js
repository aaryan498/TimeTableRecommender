import asyncHandler from "../utils/asyncHandler.js";
import axios from "axios";

export const chatBot = asyncHandler(async (req, res) => {
  const { message } = req.body;

  if (!message || message.trim() === "") {
    return res.status(400).json({ 
      success: false,
      reply: "Please provide a valid message." 
    });
  }

  try {
    const prompt = `You are the official AI assistant for Synchron and AiCrona platform.

ABOUT SYNCHRON & AICRONA:
- Synchron is the development team that created AiCrona
- AiCrona is an AI-powered university timetable scheduling platform
- Our mission is to simplify academic scheduling through artificial intelligence

PLATFORM WORKFLOW & FEATURES:
1. Organization Registration & Setup
   - Educational institutions register and create admin accounts
   - Admin accesses the Institution Dashboard after login
   - Secure authentication and role-based access control

2. Academic Data Management
   - In Academic Data Tab, admin selects: Course, Year, Semester for timetable generation
   - Admin can add/manage: Courses, Years, Semesters, Subjects, Faculty, Sections
   - Data Taker feature allows manual entry or PDF upload for bulk data
   - System auto-parses PDF content and auto-fills information fields

3. AI Timetable Generation
   - Admin clicks 'Generate And Save Timetable' for specific course-year-semester
   - AiCrona's AI engine generates 3 optimized timetable variants
   - Optimization considers: Faculty availability, room capacity, subject priorities
   - Each variant shows section → faculty → room level details

4. Timetable Management
   - Admin reviews and compares all 3 timetable variants
   - Selects and approves the most suitable variant
   - Approved timetable becomes the official schedule
   - All generated timetables accessible in Timetable Manager tab

RESPONSE REQUIREMENTS:
- Provide clear, natural English responses without any character duplication
- Do NOT duplicate characters or add extra spaces between words
- Use normal spacing and proper punctuation
- Respond in a conversational but professional tone
- If you don't know something, admit it and guide the user to contact support
- Keep responses concise but helpful

USER QUESTION: ${message.trim()}

Please provide a helpful response that directly answers the user's question.`;

    const url = process.env.GEMINI_URL;
    
    if (!url) {
      console.error("Gemini URL not configured");
      return res.status(500).json({
        success: false,
        reply: "Service configuration error. Please contact support.",
        error: "GEMINI_URL environment variable not set"
      });
    }

    console.log("Sending request to Gemini API...");
    console.log("User Message:", message);

    const response = await axios.post(
      url,
      {
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          temperature: 0.3, // Lower temperature for more consistent output
          topK: 20,
          topP: 0.8,
          maxOutputTokens: 1024,
        },
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          }
        ]
      },
      {
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Accept-Charset': 'utf-8'
        }
      }
    );

    console.log("Gemini API Response Status:", response.status);

    // Enhanced response extraction with multiple fallbacks
    let reply = extractReply(response.data);
    
    if (!reply) {
      throw new Error("Could not extract valid reply from Gemini response");
    }

    console.log("Raw Reply from Gemini:", reply);
    console.log("Reply Length:", reply.length);

    // Aggressive cleaning for the duplication issue
    const cleanReply = aggressiveCleanResponse(reply);
    
    console.log("After Cleaning:", cleanReply);

    if (!cleanReply || cleanReply.trim().length < 5) {
      throw new Error("Cleaned response is empty or too short");
    }

    res.json({ 
      success: true,
      reply: cleanReply,
      metadata: {
        originalLength: reply.length,
        cleanedLength: cleanReply.length,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err) {
    console.error("ChatBot Error:", {
      message: err.message,
      status: err.response?.status,
      data: err.response?.data
    });

    let userMessage = "I'm experiencing technical difficulties. Please try again in a moment.";
    
    if (err.response?.status === 429) {
      userMessage = "I'm currently processing too many requests. Please wait a moment and try again.";
    } else if (err.response?.status === 400) {
      userMessage = "There was an issue with the request. Please check your input and try again.";
    } else if (err.code === 'ECONNABORTED') {
      userMessage = "The request took too long to process. Please try again.";
    }

    res.status(500).json({ 
      success: false,
      reply: userMessage
    });
  }
});

// Function to extract reply from various Gemini response formats
function extractReply(data) {
  if (!data) return null;

  // Try multiple response formats
  if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
    return data.candidates[0].content.parts[0].text;
  }
  
  if (data.choices?.[0]?.message?.content) {
    return data.choices[0].message.content;
  }
  
  if (data.text) {
    return data.text;
  }
  
  if (data.response) {
    return data.response;
  }

  // Log unexpected format for debugging
  console.log("Unexpected response format:", JSON.stringify(data, null, 2));
  return null;
}

// Aggressive cleaning function for the character duplication issue
function aggressiveCleanResponse(text) {
  if (!text || typeof text !== "string") {
    return "I apologize, but I'm having trouble generating a response. Please try again.";
  }

  let cleaned = text;

  // Remove 'undefined' strings
  cleaned = cleaned.replace(/undefined/gi, '');
  
  // Fix the specific duplication pattern: "hheelllloo" -> "hello"
  // This pattern appears to be doubling each character
  cleaned = cleaned.replace(/([a-zA-Z])\1/g, '$1');
  
  // Fix triple or more duplicates
  cleaned = cleaned.replace(/([a-zA-Z])\1+/g, '$1');
  
  // Fix duplicated words with spaces: "hello  hello" -> "hello"
  cleaned = cleaned.replace(/(\b\w+\b)\s+\1/g, '$1');
  
  // Fix the specific pattern you're seeing: "yy n n cc hh rr oo nn" -> "synchron"
  // This handles spaces between duplicated characters
  cleaned = cleaned.replace(/([a-zA-Z])\s+\1/g, '$1');
  
  // Remove extra spaces (multiple spaces to single space)
  cleaned = cleaned.replace(/\s+/g, ' ');
  
  // Remove any remaining special characters or artifacts
  cleaned = cleaned.replace(/[��]/g, '');
  cleaned = cleaned.replace(/[^\x20-\x7E\u0900-\u097F]/g, ''); // Keep only printable chars and Devanagari
  
  // Ensure proper sentence structure
  cleaned = cleaned.replace(/\s+\./g, '.');
  cleaned = cleaned.replace(/\s+,/g, ',');
  cleaned = cleaned.replace(/\s+!/g, '!');
  cleaned = cleaned.replace(/\s+\?/g, '?');
  
  // Trim and capitalize first letter
  cleaned = cleaned.trim();
  if (cleaned.length > 0) {
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
  }

  // Final validation - if still corrupted, provide fallback
  // if (isStillCorrupted(cleaned)) {
  //   return "I apologize, but I'm experiencing technical issues with my response generation. Please try rephrasing your question or contact support if the problem continues.";
  // }

  return cleaned;
}

// Check if text still has the duplication corruption
function isStillCorrupted(text) {
  if (!text) return true;
  
  // Check for excessive single character repetition
  const corruptionPattern = /(.)\1\1/; // Three or more of the same character in a row
  const spacePattern = /([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])/; // Spaces between single chars
  
  return corruptionPattern.test(text) || spacePattern.test(text);
}