import asyncHandler from "../utils/asyncHandler.js";
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Use correct model name
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

export const chatBot = asyncHandler(async (req, res) => {
  const { message } = req.body;

  try {
    const prompt =
      "You are the official assistant for the entire Synchron ecosystem.\n" +
      "Synchron is the development team of AiCrona.\n" +
      "AiCrona is the AI-powered university timetable scheduler built by Synchron.\n\n" +
      "Platform usage workflow:\n" +
      "- Organisation registers and logs in.\n" +
      "Admin go to the Institution Dashboard " +
      "In the Academic Data Tab ,Selects the course,Year,semester he required to generate timetable for" +
      "- Admin adds courses, years, semesters, subjects, faculty, and sections.in the Data taker if data is too long he can upload a pdf that consist of info \n" +
      "System will auto parse and display data,it will autfill the info feilds" +
      "- Admin clicks 'Generate And Save Timetable' for a specific course-year-semester.\n" +
      "- AiCrona returns 3 optimized timetable variants.\n" +
      "- Admin views these variants at section → faculty → room level.\n" +
      "- Admin selects and approves one variant, which gets saved as the official timetable.You can view in timetable manager tab your generated timetables\n\n" +
      "You explain any part of the platform clearly, help troubleshoot steps, guide the user through features, and assist with tasks like generating PDFs when they provide information.\n\n" +
      "User message: " + message;

    const result = await model.generateContent(prompt);
    const reply = result.response.text();

    res.json({ reply });
  } catch (err) {
    console.error(err);
    res.json({ reply: "Something went wrong. Please try again." });
  }
});
