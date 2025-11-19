import { z } from "zod";
import { FacultyTimetable } from "../models/facultyTimetable.model.js";
import { OrganisationData } from "../models/organisationData.model.js";
import { SectionTimetable } from "../models/sectionTimetable.model.js";
import { Timetable } from "../models/timetable.model.js";
import asyncHandler from "../utils/asyncHandler.js";
import ApiResponse from "../utils/apiResponse.js";
import ApiError from "../utils/apiError.js";

// No-emoji validator
const noEmoji = (s) => {
  try {
    return !/\p{Extended_Pictographic}/u.test(s);
  } catch {
    return true;
  }
};

const stringField = z.string().min(1).max(200).refine(noEmoji, "No emoji allowed");

const saveTimetableSchema = z.object({
  college_info: z.any(),
  time_slots: z.any(),
  departments: z.any(),
  subjects: z.any(),
  labs: z.any(),
  faculty: z.any(),
  rooms: z.any(),
  constraints: z.any(),
  special_requirements: z.any(),
  genetic_algorithm_params: z.any(),
});

const validate = (schema, data) => {
  const r = schema.safeParse(data);
  if (!r.success) {
    const msg = r.error.errors[0].message;
    const err = new Error(msg);
    err.statusCode = 400;
    throw err;
  }
  return r.data;
};

export const resetOrganisationData = asyncHandler(async(req,res)=>{

  const organisationId = req.organisation._id;
    
await Promise.all([
  OrganisationData.deleteOne({ organisationId }),
  SectionTimetable.deleteMany({ organisationId }),
  FacultyTimetable.deleteMany({ organisationId })
]);


return res.status(200).json(
  new ApiResponse(204,{},"Organisation Data has been reset successfully")
)

})
export const saveTimetable = async (req, res) => {
  try {
    const organisationId = req.organisation?._id;
    if (!organisationId) return res.status(401).json({ message: "Login first" });
   const {course,year,semester} = req.query;
   
   if(!course || course.trim()==="" || !year || year.trim()==="" || !semester || semester.trim()==="") 
   {
   throw new ApiError(400,"Year and Course are not specified");
   }

    const body = validate(saveTimetableSchema, req.body);

    const updateData = { organisationId, year:year.toLowerCase(),course:course.toLowerCase(),semester:semester.trim().toLowerCase(),...body };

    const timetable = await OrganisationData.findOneAndUpdate(
      { organisationId,
        course,
        year,
        semester

       },
      { $set: updateData },
      { new: true, upsert: true }
    );

    await Promise.all([
      SectionTimetable.deleteMany({ organisationId }),
      FacultyTimetable.deleteMany({ organisationId }),
    ]);

    res.status(201).json({ message: "Data saved/updated successfully", timetable });
  } catch (error) {
    res.status(error.statusCode || 500).json({ message: "Error saving timetable", error: error.message });
  }
};

export const getPreviousSavedData = async (req, res) => {
  try {
    const organisationId = req.organisation?._id;
    const {course,year,semester} = req.query;
    if(!course || !year)
    {
      throw new ApiError(400,"Course or year is not specified");
    }
    if (!organisationId) return res.status(401).json({ message: "Login first" });

    const previousData = await OrganisationData.findOne({ organisationId,course:course.trim().toLowerCase(),year:year.trim().toLowerCase(),semester:semester.trim().toLowerCase() });
    if (!previousData) return res.status(404).json({ message: "No previously saved timetable found" });

    const { organisationId: _skip,course:skip1,year:skip3, ...rest } = previousData.toObject();

    res.status(200).json({ message: "Previous timetable fetched successfully", data: rest });
  } catch (error) {
    res.status(500).json({ message: "Error fetching previous timetable", error: error.message });
  }
};

export const getLatestTimetable = async (req, res) => {
  try {
    const timetable = await OrganisationData.findOne().sort({ createdAt: -1 });
    if (!timetable) return res.status(404).json({ message: "No timetable found" });
    res.status(200).json(timetable);
  } catch (error) {
    res.status(500).json({ message: "Error fetching timetable", error: error.message });
  }
};

export const getAllTimetables = async (req, res) => {
  try {
    const timetables = await Timetable.find().sort({ createdAt: -1 });
    res.status(200).json(timetables);
  } catch (error) {
    res.status(500).json({ message: "Error fetching timetables", error: error.message });
  }
};
