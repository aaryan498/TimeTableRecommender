import { FacultyTimetable } from "../models/facultyTimetable.model.js";
import { OrganisationData } from "../models/organisationData.model.js";
import { SectionTimetable } from "../models/sectionTimetable.model.js";


// Save new timetable
export const saveTimetable = async (req, res) => {
  try {
    const organisationId = req.organisation?._id;
    if (!organisationId) {
      return res.status(400).json({ message: "Organisation not found in request" });
    }

    console.log("Incoming timetable data:", req.body);

    const timetable = await OrganisationData.findOneAndUpdate(
      { organisationId },
      { $set: { organisationId, ...req.body } },
      { new: true, upsert: true }
    );

    if (!timetable) {
      throw new Error("Failed to save or update timetable");
    }

    console.log("✅ Timetable document saved/updated:", timetable._id);

    // ✅ Delete previous section and faculty timetables in parallel
    const [sectionsDeleted, facultyDeleted] = await Promise.all([
      SectionTimetable.deleteMany({ organisationId }),
      FacultyTimetable.deleteMany({ organisationId }),
    ]);

    console.log(
      `🧹 Cleared old timetables — Sections: ${sectionsDeleted.deletedCount}, Faculty: ${facultyDeleted.deletedCount}`
    );

    // ✅ Respond success
    res.status(201).json({
      message: "Timetable saved/updated successfully",
      timetable,
      deleted: {
        sections: sectionsDeleted.deletedCount,
        faculty: facultyDeleted.deletedCount,
      },
    });

  } catch (error) {
    console.error("❌ Error saving timetable:", error);
    res.status(500).json({
      message: "Error saving timetable",
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
