import { FacultyTimetable } from "../models/facultyTimetable.model.js";
import { OrganisationData } from "../models/organisationData.model.js";
import { SectionTimetable } from "../models/sectionTimetable.model.js";


// Save new timetable

export const saveTimetable = async (req, res) => {
  try {
     
    const organisationId = req.organisation._id ;
     
   console.log("Here is the data coming",req.body)

    // console.log("Here is the req body",req.body)
    
    // Use organisation+semester+section as unique combination (you can change based on your needs)
    const timetable = await OrganisationData.findOneAndUpdate(
      { organisationId }, 
      { $set: {organisationId,...req.body} }, // update fields
      { new: true, upsert: true } // create if not exists
    );
   console.log("Here is the saved document",timetable)
  // Also empty the faculty and section timetables stored previously

    const sectionsDeleted = await SectionTimetable.deleteMany({organisationId});
    const facultyDeleted = await FacultyTimetable.deleteMany({organisationId})
 

    console.log("Pre Saved timetables are empty now")




    res.status(201).json({
      message: "Timetable saved/updated successfully",
      timetable,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      message: "Error saving timetable",
      error,
    });
  }
};

// Get latest timetable
export const getLatestTimetable = async (req, res) => {
  try {
    const timetable = await OrganisationData.findOne().sort({ createdAt: -1 });
    if (!timetable) return res.status(404).json({ message: 'No timetable found' });
    res.status(200).json(timetable);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching timetable', error });
  }
};

// Get all timetables
export const getAllTimetables = async (req, res) => {
  try {
    const timetables = await Timetable.find().sort({ createdAt: -1 });
    res.status(200).json(timetables);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching timetables', error });
  }
};
