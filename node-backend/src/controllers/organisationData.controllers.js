import { FacultyTimetable } from "../models/facultyTimetable.model.js";
import { OrganisationData } from "../models/organisationData.model.js";
import { SectionTimetable } from "../models/sectionTimetable.model.js";


export const saveTimetable = async (req, res) => {
  try {
    const organisationId = req.organisation?._id;
    if (!organisationId) {
      return res.status(401).json({ message: "Login first" });
    }

    console.log("Incoming timetable data:", req.body);

    const {
      college_info,
      time_slots,
      departments,
      subjects,
      labs,
      faculty,
      rooms,
      constraints,
      special_requirements,
      genetic_algorithm_params
    } = req.body;

    const updateData = {
      organisationId,
      college_info,
      time_slots,
      departments,
      subjects,
      labs,
      faculty,
      rooms,
      constraints,
      special_requirements,
      genetic_algorithm_params,
    };

    const timetable = await OrganisationData.findOneAndUpdate(
      { organisationId },
      { $set: updateData },
      { new: true, upsert: true }
    );

    await Promise.all([
      SectionTimetable.deleteMany({ organisationId }),
      FacultyTimetable.deleteMany({ organisationId }),
    ]);

    res.status(201).json({
      message: "Timetable saved/updated successfully",
      timetable,
    });
  } catch (error) {
    console.error("Error saving timetable:", error);
    res.status(500).json({
      message: "Error saving timetable",
      error: error.message || error,
    });
  }
};


export const getPreviousSavedData = async (req, res) => {
  try {
    const organisationId = req.organisation?._id;
    if (!organisationId) {
      return res.status(401).json({ message: "Login first" });
    }

    const previousData = await OrganisationData.findOne({ organisationId });

    if (!previousData) {
      return res.status(404).json({
        message: "No previously saved timetable found",
      });
    }

    const { organisationId: _orgId, ...responseData } = previousData.toObject();

    res.status(200).json({
      message: "Previous timetable data fetched successfully",
      data: responseData,
    });
  } catch (error) {
    console.error("Error fetching previous timetable:", error);
    res.status(500).json({
      message: "Error fetching previous timetable",
      error: error.message || error,
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
