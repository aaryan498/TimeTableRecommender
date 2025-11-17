import axios from "axios";
import { SectionTimetable } from "../models/sectionTimetable.model.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiResponse from "../utils/apiResponse.js";
import ApiError from "../utils/apiError.js";


export const downloadAllSectionTimetables = async (req, res) => {
  try {
    // 1) Try DB first
     console.log("I had been")

    let docs = await SectionTimetable.find().lean();

    if (!docs || docs.length === 0) {
      console.log("No timetables in DB. Fetching from API...");
      const FLASK_URL = "http://127.0.0.1:5000";
      const apiResp = await axios.get(`${FLASK_URL}/api/timetables/sections`);
      const sectionsObj = apiResp?.data;

      if (!sectionsObj || Object.keys(sectionsObj).length === 0) {
        return res.status(404).json({ success: false, message: "No timetables found from API" });
      }

      const sectionsArr = Object.values(sectionsObj);

      // Save to DB
      const ops = sectionsArr.map((sec) => ({
        updateOne: {
          filter: { section_id: sec.section_id },
          update: {
            $set: {
              section_name: sec.section_name,
              semester: sec.semester,
              specialization: sec.specialization || "",
              periods: sec.periods || {},
              timetable: sec.timetable || {},
            },
          },
          upsert: true,
        },
      }));
      await SectionTimetable.bulkWrite(ops);

    }

    
  } catch (err) {
    console.error("Error in downloadAllSectionTimetables:", err);
    res.status(500).json({ success: false, message: "Server error generating timetables PDF" });
  }
};


// Update Manual



// Update AI based
// Update a specific slot in a section timetable
export const updateTimetableSlot = asyncHandler(async (req, res) => {
  try {
    const { sectionId, day, period } = req.params;
    const updateData = req.body;

    // Validate required fields
    if (!sectionId || !day || !period) {
      return res.status(400).json(
        new ApiResponse(400, null, "Section ID, day, and period are required")
      );
    }

    // Find the section timetable
    const sectionTimetable = await SectionTimetable.findOne({
      $or: [
        { section_id: sectionId },
        { section_name: sectionId }
      ]
    });

    if (!sectionTimetable) {
      return res.status(404).json(
        new ApiResponse(404, null, "Section timetable not found")
      );
    }

    // Initialize timetable object if it doesn't exist
    if (!sectionTimetable.timetable) {
      sectionTimetable.timetable = {};
    }

    // Initialize day object if it doesn't exist
    if (!sectionTimetable.timetable[day]) {
      sectionTimetable.timetable[day] = {};
    }

    // Handle FREE slot (clear the slot)
    if (updateData.subject === "FREE" || updateData === "FREE") {
      sectionTimetable.timetable[day][period] = "FREE";
    } else {
      // Update or create the slot with new data
      sectionTimetable.timetable[day][period] = {
        subject: updateData.subject || "",
        section: updateData.section || "",
        room: updateData.room || "",
        type: updateData.type || "Lecture",
        // Preserve existing data if not provided
        ...(sectionTimetable.timetable[day][period] && 
            typeof sectionTimetable.timetable[day][period] === 'object' ? 
            sectionTimetable.timetable[day][period] : {})
      };
    }

    // Update last modified timestamp
    sectionTimetable.lastUpdated = new Date();

    // Save the updated timetable
    const updatedTimetable = await sectionTimetable.save();

    // Convert to lean object and remove MongoDB fields
    const cleanTimetable = updatedTimetable.toObject();
    delete cleanTimetable._id;
    delete cleanTimetable.__v;

    return res.status(200).json(
      new ApiResponse(200, { data: cleanTimetable }, "Timetable slot updated successfully")
    );

  } catch (error) {
    console.error("Error updating timetable slot:", error);
    return res.status(500).json(
      new ApiResponse(500, null, "Internal server error while updating timetable")
    );
  }
});
export const updateMultipleTimetableSlots = asyncHandler(async (req, res) => {
  try {
    const { sectionId } = req.params;
    const { updates } = req.body;

    // Validate required fields
    if (!sectionId || !updates || !Array.isArray(updates)) {
      return res.status(400).json(
        new ApiResponse(400, null, "Section ID and updates array are required")
      );
    }

    // Find the section timetable
    const sectionTimetable = await SectionTimetable.findOne({
      $or: [
        { section_id: sectionId },
        { section_name: sectionId }
      ]
    });

    if (!sectionTimetable) {
      return res.status(404).json(
        new ApiResponse(404, null, "Section timetable not found")
      );
    }

    // Initialize timetable object if it doesn't exist
    if (!sectionTimetable.timetable) {
      sectionTimetable.timetable = {};
    }

    // Process each update
    updates.forEach(update => {
      const { day, period, data } = update;

      if (!day || !period) {
        throw new Error("Each update must include day and period");
      }

      // Initialize day object if it doesn't exist
      if (!sectionTimetable.timetable[day]) {
        sectionTimetable.timetable[day] = {};
      }

      // Handle FREE slot (clear the slot)
      if (data === "FREE" || (data && data.subject === "FREE")) {
        sectionTimetable.timetable[day][period] = "FREE";
      } else if (data && typeof data === 'object') {
        // Update or create the slot with new data
        sectionTimetable.timetable[day][period] = {
          subject: data.subject || "",
          section: data.section || "",
          room: data.room || "",
          type: data.type || "Lecture",
          // Preserve existing data if not provided
          ...(sectionTimetable.timetable[day][period] && 
              typeof sectionTimetable.timetable[day][period] === 'object' ? 
              sectionTimetable.timetable[day][period] : {})
        };
      }
    });

    // Update last modified timestamp
    sectionTimetable.lastUpdated = new Date();

    // Save the updated timetable
    const updatedTimetable = await sectionTimetable.save();

    // Convert to lean object and remove MongoDB fields
    const cleanTimetable = updatedTimetable.toObject();
    delete cleanTimetable._id;
    delete cleanTimetable.__v;

    return res.status(200).json(
      new ApiResponse(200, { data: cleanTimetable }, "Multiple timetable slots updated successfully")
    );

  } catch (error) {
    console.error("Error updating multiple timetable slots:", error);
    return res.status(500).json(
      new ApiResponse(500, null, "Internal server error while updating timetable")
    );
  }
});
export const updateSectionTimetable = asyncHandler(async (req, res) => {
  try {

     console.log("Here is the incoming body",req.body)
    const { section_id } = req.body;
  const updateData = req.body;

    

    if (!section_id) {
      return res.status(400).json(
        new ApiResponse(400, null, "Section ID is required")
      );
    }
     const existingSection = await SectionTimetable.findOne({ section_id });
        if (!existingSection) {
          throw new ApiError(404, `Section with ID ${section_id} not found`);
        }



    const updatedSection = await FacultyTimetable.findOneAndUpdate(
      { section_id },
      updateData,
      {
        new: true,
        runValidators: true, 
        context: 'query' 
      }
    ).lean();
    if (!updatedSection) {
      return res.status(404).json(
        new ApiResponse(404, null, "Section timetable not found")
      );
    }


    return res.status(200).json(
      new ApiResponse(200, updatedSection, "Section timetable updated successfully")
    );

  } catch (error) {
  console.error("Error in updateSectionTimetable:", error);

    if (error instanceof ApiError) {
      throw error;
    }

    if (error.name === 'ValidationError') {
      const errors = Object.values(error.errors).map(err => err.message);
      throw new ApiError(400, `Validation error: ${errors.join(', ')}`);
    }
    if (error.name === 'CastError') {
      throw new ApiError(400, "Invalid faculty ID format");
    }

    throw new ApiError(500, "Internal server error while updating section timetable");
  }
  
});
export const clearDaySlots = asyncHandler(async (req, res) => {
  try {
    const { sectionId, day } = req.params;

    // Validate required fields
    if (!sectionId || !day) {
      return res.status(400).json(
        new ApiResponse(400, null, "Section ID and day are required")
      );
    }

    // Find the section timetable
    const sectionTimetable = await SectionTimetable.findOne({
      $or: [
        { section_id: sectionId },
        { section_name: sectionId }
      ]
    });

    if (!sectionTimetable) {
      return res.status(404).json(
        new ApiResponse(404, null, "Section timetable not found")
      );
    }

    // Initialize timetable object if it doesn't exist
    if (!sectionTimetable.timetable) {
      sectionTimetable.timetable = {};
    }

    // Clear all slots for the specified day
    sectionTimetable.timetable[day] = {};

    // Update last modified timestamp
    sectionTimetable.lastUpdated = new Date();

    // Save the updated timetable
    const updatedTimetable = await sectionTimetable.save();

    // Convert to lean object and remove MongoDB fields
    const cleanTimetable = updatedTimetable.toObject();
    delete cleanTimetable._id;
    delete cleanTimetable.__v;

    return res.status(200).json(
      new ApiResponse(200, { data: cleanTimetable }, `All slots cleared for ${day}`)
    );

  } catch (error) {
    console.error("Error clearing day slots:", error);
    return res.status(500).json(
      new ApiResponse(500, null, "Internal server error while clearing day slots")
    );
  }
});
export const getUpdateHistory = asyncHandler(async (req, res) => {
  try {
    const { sectionId } = req.params;

    // Validate required fields
    if (!sectionId) {
      return res.status(400).json(
        new ApiResponse(400, null, "Section ID is required")
      );
    }

    const sectionTimetable = await SectionTimetable.findOne({
      $or: [
        { section_id: sectionId },
        { section_name: sectionId }
      ]
    }).select('lastUpdated section_name section_id');

    if (!sectionTimetable) {
      return res.status(404).json(
        new ApiResponse(404, null, "Section timetable not found")
      );
    }

    return res.status(200).json(
      new ApiResponse(200, {
        data: {
          section_name: sectionTimetable.section_name,
          section_id: sectionTimetable.section_id,
          lastUpdated: sectionTimetable.lastUpdated
        }
      }, "Update history fetched successfully")
    );

  } catch (error) {
    console.error("Error fetching update history:", error);
    return res.status(500).json(
      new ApiResponse(500, null, "Internal server error while fetching update history")
    );
  }
});