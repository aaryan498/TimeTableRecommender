import asyncHandler from "../utils/asyncHandler.js";
import axios from "axios";
import ApiError from "../utils/apiError.js";
import ApiResponse from "../utils/apiResponse.js";
import { FacultyTimetable } from "../models/facultyTimetable.model.js";
import path from "path";
import fs from "fs"
import { SectionTimetable } from "../models/sectionTimetable.model.js";
import FormData from "form-data";
import { OrganisationData } from "../models/organisationData.model.js";

import { nanoid } from "nanoid";

const { FLASK_URL } = process.env



export const getTheDynamicResult = asyncHandler(async (req, res) => {


  console.log("The result is going to be send")


  const flaskRes = await axios.post("")



})
let c = 1;
export const generateByGivingData = asyncHandler(async (req, res) => {

  console.log("I am working manual data", c);
  c++;


  const parsed_config = req.body;

  // console.log( "I am the coming parsed config",  parsed_config)


  if (!parsed_config) {
    throw new ApiError(404, "Please provide the input for the generation")
  }


  // const sendingTheData = await axios.post(`http://localhost:8080/api/timetable/sendData`,parsed_config,{
  //   withCredentials:true
  // })



  // if (!sendingTheData) {
  //   throw new ApiError(500, "Sorry our model is busy")
  // }


  return res.status(202).json(
    new ApiResponse(202, {}, "Sent for Generation")
  )
})

export const getInfoPdf = async (req, res) => {
  try {
    const pdfPath = req.file.path;


    const formData = new FormData();
    formData.append("file", fs.createReadStream(pdfPath));
    // formData.append("college_name", req.body.college_name || "");
    // formData.append("session", req.body.session || "");

    const flaskRes = await axios.post(
      `${FLASK_URL}/api/parse-timetable`,
      formData,
      {
        headers: formData.getHeaders(),
      }
    );

    fs.unlink(pdfPath, (err) => {
      if (err) {
        console.log("Error while deleting the pdf", err)
      }
    });

    res.json({
      success: true,
      parsedConfig: flaskRes.data.data,
      extractionInfo: flaskRes.data.extraction_info || {},
    });
  } catch (err) {

    fs.unlink(pdfPath, (err) => {
      if (err) {
        console.log("Error while deleting the pdf", err)
      }
    });
    console.error("PDF upload error:", err.response?.data || err.message);
    res.status(500).json({ success: false, error: "PDF processing failed" });
  }
};





/**
 * Trigger timetable generation in Flask (background)
 */
export const startTimeTableCreation = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;

  if (!organisationId) {
    throw new ApiError(400, "Login First");
  }

  if (!course?.trim() || !year?.trim() || !semester?.trim()) {
    throw new ApiError(400, "Course, Year and Semester are required");
  }

  const c = course.trim().toLowerCase();
  const y = year.trim().toLowerCase();
  const s = semester.trim().toLowerCase();

  const organisationData = await OrganisationData.findOne({
    organisationId,
    course: c,
    year: y,
    semester: s,
  });

  if (!organisationData) {
    return res.status(404).json({
      success: false,
      statusCode: 404,
      message: "Organisation data not found",
      data: null,
    });
  }

  const transformedData = {
    college_info: {
      name: organisationData.college_info?.name || "",
      session: organisationData.college_info?.session || "",
      effective_date: organisationData.college_info?.effective_date
        ? new Date(organisationData.college_info.effective_date)
            .toISOString()
            .split("T")[0]
        : "",
    },
    time_slots: {
      periods: (organisationData.time_slots?.periods || []).map((p) => ({
        id: p.id,
        start_time: p.start_time,
        end_time: p.end_time,
      })),
      working_days: organisationData.time_slots?.working_days || [],
      break_periods: organisationData.time_slots?.break_periods || [],
      lunch_period: organisationData.time_slots?.lunch_period || null,
      mentorship_period: organisationData.time_slots?.mentorship_period || null,
    },
    departments: (organisationData.departments || []).map((dept) => ({
      dept_id: dept.dept_id,
      name: dept.name,
      sections: (dept.sections || []).map((sec) => ({
        section_id: sec.section_id,
        name: sec.name,
        semester: sec.semester,
        year: sec.year,
        room: sec.room,
        student_count: sec.student_count,
        coordinator: sec.coordinator,
        specialization: sec.specialization || "",
      })),
    })),
    subjects: organisationData.subjects || [],
    labs: organisationData.labs || [],
    faculty: organisationData.faculty || [],
    rooms: organisationData.rooms || [],
    constraints: organisationData.constraints || {},
    special_requirements: organisationData.special_requirements || {},
    genetic_algorithm_params: organisationData.genetic_algorithm_params || {},
  };


  console.log("Here is the going to send to model")

  const response = await axios.post(`${FLASK_URL}/api/generate`, transformedData, {
    withCredentials: true,
  });

  if (!response?.data) {
    throw new ApiError(500, "Failed to start generation");
  }

  const facultyDataObj = response.data.data.faculty;
  const facultyIds = Object.keys(facultyDataObj);

  for (const key of facultyIds) {
    const facultyData = facultyDataObj[key];

    await FacultyTimetable.findOneAndUpdate(
      {
        organisationId,
        course: c,
        year: y,
        semester: s,
        faculty_id: facultyData.faculty_id, // now uses provided id
      },
      {
        ...facultyData,
        organisationId,
        course: c,
        year: y,
        semester: s,
      },
      { new: true, upsert: true }
    );
  }

  const sectionsObj = response.data.data.sections;
  const sectionsArr = Object.values(sectionsObj);

  const ops = sectionsArr.map((sec) => ({
    updateOne: {
      filter: {
        organisationId,
        course: c,
        year: y,
        semester: s,
        section_id: sec.section_id,
      },
      update: {
        $set: {
          organisationId,
          course: c,
          year: y,
          semester: s,
          section_id: sec.section_id,
          section_name: sec.section_name,
          specialization: sec.specialization || "",
          periods: sec.periods || {},
          timetable: sec.timetable || {},
        },
      },
      upsert: true,
    },
  }));

  await SectionTimetable.bulkWrite(ops, { ordered: false });

  return res.json(
    new ApiResponse(200, {transformedData,response}, "Timetable generated and saved successfully")
  );
});






export const checkGenerationStatus = asyncHandler(async (req, res) => {
  const response = await axios.get(`${FLASK_URL}/api/status`);
  if (!response || !response.data) throw new ApiError(500, "Failed to fetch status");

  return res.json(new ApiResponse(200, response.data, "Status fetched"));
});



export const getSectionTimeTablesDb = asyncHandler(async (req, res) => {
  try {
    const organisationId = req.organisation?._id;

    if (!organisationId) {
      throw new ApiError(401, "Login first");
    }

    const docs = await SectionTimetable.find({ organisationId })
      .select("-organisationId")
      .lean();

    if (!docs || docs.length === 0) {
      throw new ApiError(400, "No timetable found for sections");
    }

    const grouped = {};

    docs.forEach(doc => {
      const { _id, __v, createdAt, updatedAt, ...clean } = doc;

      const { course, year, semester, section_id } = clean;

      if (!grouped[course]) grouped[course] = {};
      if (!grouped[course][year]) grouped[course][year] = {};
      if (!grouped[course][year][semester]) grouped[course][year][semester] = {};

      grouped[course][year][semester][section_id] = clean;
    });

    return res.json(
      new ApiResponse(200, grouped, "Section timetables fetched successfully")
    );

  } catch (error) {
    console.error("Unexpected error in getSectionTimeTables:", error);

    return res
      .status(500)
      .json(new ApiError(500, "Internal server error"));
  }
});





export const getSingleSectionTimeTable = asyncHandler(async (req, res) => {
  const { section_id } = req.params;
  const response = await axios.get(`${FLASK_URL}/api/timetables/sections/${section_id}`);
  if (!response || !response.data) throw new ApiError(404, "Section timetable not found");

  return res.json(new ApiResponse(200, response, `Section ${section_id} timetable fetched`));
});


export const getFacultyTimeTables = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;

  if (!organisationId) {
    throw new ApiError(401, "Login first");
  }

  const docs = await FacultyTimetable.find({ organisationId }).lean();

  if (!docs || docs.length === 0) {
    throw new ApiError(400, "No timetables found");
  }

  const grouped = {};

  docs.forEach(doc => {
    const {
      _id,
      __v,
      createdAt,
      updatedAt,
      organisationId: orgId,
      ...clean
    } = doc;

    const { course, year, semester, faculty_id } = clean;

    if (!grouped[course]) grouped[course] = {};
    if (!grouped[course][year]) grouped[course][year] = {};
    if (!grouped[course][year][semester]) grouped[course][year][semester] = {};

    grouped[course][year][semester][faculty_id] = clean;
  });

  return res.status(200).json(
    new ApiResponse(200, grouped, "Faculty timetables fetched successfully")
  );
});


const getAllGeneratedSectionTimeTables = asyncHandler(async (req, res) => {

  const organisationId = req.organisation_id;


  const sectionTimeTables = await SectionTimetable.find({ organisationId });


  if (sectionTimeTables.length === 0) {
    throw new ApiError(400, "No section TimTables yet")
  }


  return res.status(200).json(
    new ApiResponse(200, sectionTimeTables, "Section TimeTable fetched successfully")
  )

})


export const updateFacultyTimetable = asyncHandler(async (req, res) => {
  console.log("I have been hit - updateFacultyTimetable");


  const organisationId = req.organisation?._id;
  const { faculty_id } = req.body;
  const updateData = req.body;

  try {

    if (!faculty_id) {
      throw new ApiError(400, "Faculty ID is required");
    }

    // Validate update data
    if (!updateData || Object.keys(updateData).length === 0) {
      throw new ApiError(400, "Update data is required");
    }

    // Check if faculty exists
    const existingFaculty = await FacultyTimetable.findOne({ faculty_id });
    if (!existingFaculty) {
      throw new ApiError(404, `Faculty with ID ${faculty_id} not found`);
    }

    // Update the faculty timetable
    const updatedFaculty = await FacultyTimetable.findOneAndUpdate(
      {
        faculty_id,
        organisationId
      },
      updateData,
      {
        new: true, // Return the updated document
        runValidators: true, // Run schema validators
        context: 'query' // Ensure validators work with update
      }
    ).lean();

    if (!updatedFaculty) {
      throw new ApiError(500, "Failed to update faculty timetable");
    }

    console.log(`Faculty timetable updated for ID: ${faculty_id}`);

    return res.status(200).json(
      new ApiResponse(200, updatedFaculty, "Faculty timetable updated successfully")
    );

  } catch (error) {
    console.error("Error in updateFacultyTimetable:", error);

    if (error instanceof ApiError) {
      throw error;
    }

    // Handle MongoDB validation errors
    if (error.name === 'ValidationError') {
      const errors = Object.values(error.errors).map(err => err.message);
      throw new ApiError(400, `Validation error: ${errors.join(', ')}`);
    }

    // Handle cast errors (invalid ID format)
    if (error.name === 'CastError') {
      throw new ApiError(400, "Invalid faculty ID format");
    }

    throw new ApiError(500, "Internal server error while updating faculty timetable");
  }
});



/**
 * Get single faculty timetable by ID
 */
export const getSingleFacultyTimeTable = asyncHandler(async (req, res) => {
  const { faculty_id } = req.params;
  const response = await axios.get(`${FLASK_URL}/api/timetables/faculty/${faculty_id}`);
  if (!response || !response.data) throw new ApiError(404, "Faculty timetable not found");

  return res.json(new ApiResponse(200, response.data, `Faculty ${faculty_id} timetable fetched`));
});

/**
 * Get detailed timetable
 */
export const getDetailedTimeTable = asyncHandler(async (req, res) => {
  const response = await axios.get(`${FLASK_URL}/api/timetables/detailed`);
  if (!response || !response.data) throw new ApiError(404, "Detailed timetable not found");

  return res.json(new ApiResponse(200, response.data, "Detailed timetable fetched"));
});



export const getFacultyTimetablesByGroup = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;

  if (!organisationId) throw new ApiError(401, "Login first");

  if (!course || !year || !semester) {
    throw new ApiError(400, "Course, year, and semester are required");
  }

  const docs = await FacultyTimetable.find({
    organisationId,
    course: course.toLowerCase(),
    year: year.toLowerCase(),
    semester: semester.toLowerCase()
  })
    .select("-_id -__v -createdAt -updatedAt -organisationId")
    .lean();

  if (!docs || docs.length === 0) {
    throw new ApiError(404, "No faculty timetables found");
  }

  // FORMAT:
  // { faculty_id: { faculty_name, department, periods, timetable, ... } }
  const result = {};

  docs.forEach(doc => {
    result[doc.faculty_id] = {
      faculty_id: doc.faculty_id,
      faculty_name: doc.faculty_name,
      department: doc.department,
      periods: doc.periods,
      timetable: doc.timetable
    };
  });

  return res.status(200).json(
    new ApiResponse(200, {
      course: course.toLowerCase(),
      year: year.toLowerCase(),
      semester: semester.toLowerCase(),
      faculty: result
    }, "Faculty timetables fetched")
  );
});
export const getSectionTimetablesByGroup = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;

  if (!organisationId) throw new ApiError(401, "Login first");

  if (!course || !year || !semester) {
    throw new ApiError(400, "Course, year, and semester are required");
  }

  const docs = await SectionTimetable.find({
    organisationId,
    course: course.toLowerCase(),
    year: year.toLowerCase(),
    semester: semester.toLowerCase()
  })
    .select("-_id -__v -createdAt -updatedAt -organisationId")
    .lean();

  if (!docs || docs.length === 0) {
    throw new ApiError(404, "No section timetables found");
  }

  // FORMAT:
  // { section_id: { section_name, periods, timetable, ... } }
  const result = {};

  docs.forEach(doc => {
    result[doc.section_id] = {
      section_id: doc.section_id,
      section_name: doc.section_name,
      specialization: doc.specialization,
      periods: doc.periods,
      timetable: doc.timetable
    };
  });

  return res.status(200).json(
    new ApiResponse(200, {
      course: course.toLowerCase(),
      year: year.toLowerCase(),
      semester: semester.toLowerCase(),
      sections: result
    }, "Section timetables fetched")
  );
});

export const getFacultyTimeTablesForSpecific = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;
  
  console.log("here are the coming things",course,year,semester)
  if (!organisationId) {
    throw new ApiError(401, "Login first");
  }

  if (!course || !year || !semester) {
    throw new ApiError(400, "Course, year, and semester are required");
  }

  const docs = await FacultyTimetable.find({
    organisationId,
    course: course.toLowerCase().trim(),
    year: year.toLowerCase().trim(),
    semester: semester.toLowerCase().trim()
  }).lean();

  console.log("Here is the docs",docs)

  if (!docs || docs.length === 0) {
    throw new ApiError(404, "No faculty timetables found");
  }

  const result = {};

  docs.forEach(doc => {
    const {
      _id,
      __v,
      createdAt,
      updatedAt,
      organisationId: org,
      ...cleanDoc
    } = doc;

    result[doc.faculty_id] = cleanDoc;
  });

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        course: course.toLowerCase(),
        year: year.toLowerCase(),
        semester: semester.toLowerCase(),
        faculty: result
      },
      "Faculty timetables fetched successfully"
    )
  );
});
export const getSectionTimeTablesForSpecific = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;

  console.log("Section params received:", course, year, semester);

  if (!organisationId) {
    throw new ApiError(401, "Login first");
  }

  if (!course || !year || !semester) {
    throw new ApiError(400, "Course, year, and semester are required");
  }

  const docs = await SectionTimetable.find({
    organisationId,
    course: course.toLowerCase().trim(),
    year: year.toLowerCase().trim(),
    semester: semester.toLowerCase().trim()
  }).lean();

  console.log("Section docs fetched:", docs);

  if (!docs || docs.length === 0) {
    throw new ApiError(404, "No section timetables found");
  }

  const result = {};

  docs.forEach(doc => {
    const {
      _id,
      __v,
      createdAt,
      updatedAt,
      organisationId: org,
      ...cleanDoc
    } = doc;

    // store using section_id as key
    result[doc.section_id] = cleanDoc;
  });

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        course: course.toLowerCase(),
        year: year.toLowerCase(),
        semester: semester.toLowerCase(),
        sections: result
      },
      "Section timetables fetched successfully"
    )
  );
});
