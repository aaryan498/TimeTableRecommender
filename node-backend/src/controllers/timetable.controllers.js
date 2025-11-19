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

  const organisationId = req.organisation?._id
  const { course, year, semester } = req.query;
  if (!organisationId) {
    throw new ApiError(400, "Login First")
  }
  if (!course || !year) {
    throw new ApiError(400, "Course or year is not specified")
  }

  const organisationData = await OrganisationData.findOne({
    organisationId,
    course: course.trim().toLowerCase(),
    year: year.trim().toLowerCase(), semester: semester.trim().toLowerCase()
  });


  if (!organisationData) {
    return res.status(404).json({
      success: false,
      statusCode: 404,
      message: "Organisation data not found",
      data: null
    });
  }

  const transformedData = {
    college_info: {
      name: organisationData.college_info?.name || "",
      session: organisationData.college_info?.session || "",
      effective_date: organisationData.college_info?.effective_date
        ? new Date(organisationData.college_info.effective_date).toISOString().split('T')[0]
        : ""
    },
    time_slots: {
      periods: (organisationData.time_slots?.periods || []).map(period => ({
        id: period.id,
        start_time: period.start_time,
        end_time: period.end_time
      })),
      working_days: organisationData.time_slots?.working_days || [],
      break_periods: organisationData.time_slots?.break_periods || [],
      lunch_period: organisationData.time_slots?.lunch_period || null,
      mentorship_period: organisationData.time_slots?.mentorship_period || null
    },
    departments: (organisationData.departments || []).map(dept => ({
      dept_id: dept.dept_id,
      name: dept.name,
      sections: (dept.sections || []).map(section => ({
        section_id: section.section_id,
        name: section.name,
        semester: typeof section.semester === 'string' ? parseInt(section.semester) || section.semester : section.semester,
        year: typeof section.year === 'string' ? parseInt(section.year) || section.year : section.year,
        room: section.room,
        student_count: section.student_count,
        coordinator: section.coordinator,
        specialization: section.specialization || ""
      }))
    })),
    subjects: (organisationData.subjects || []).map(subject => ({
      subject_id: subject.subject_id,
      name: subject.name,
      type: subject.type,
      credits: subject.credits,
      lectures_per_week: subject.lectures_per_week,
      semester: subject.semester,
      departments: subject.departments || [],
      min_classes_per_week: subject.min_classes_per_week,
      max_classes_per_day: subject.max_classes_per_day,
      tutorial_sessions: subject.tutorial_sessions || 0,
      specialization: subject.specialization || "",
      flexible_timing: subject.flexible_timing || false
    })),
    labs: (organisationData.labs || []).map(lab => ({
      lab_id: lab.lab_id,
      name: lab.name,
      type: lab.type,
      credits: lab.credits,
      sessions_per_week: lab.sessions_per_week,
      duration_hours: lab.duration_hours,
      semester: lab.semester,
      departments: lab.departments || [],
      lab_rooms: lab.lab_rooms || [],
      specialization: lab.specialization || ""
    })),
    faculty: (organisationData.faculty || []).map(faculty => ({
      faculty_id: faculty.faculty_id,
      name: faculty.name,
      department: faculty.department,
      designation: faculty.designation,
      subjects: faculty.subjects || [],
      max_hours_per_week: faculty.max_hours_per_week,
      avg_leaves_per_month: faculty.avg_leaves_per_month,
      preferred_time_slots: faculty.preferred_time_slots || []
    })),
    rooms: (organisationData.rooms || []).map(room => ({
      room_id: room.room_id,
      name: room.name,
      type: room.type,
      capacity: room.capacity,
      department: room.department || "",
      equipment: room.equipment || []
    })),
    constraints: {
      hard_constraints: {
        no_faculty_clash: organisationData.constraints?.hard_constraints?.no_faculty_clash || true,
        no_room_clash: organisationData.constraints?.hard_constraints?.no_room_clash || true,
        no_section_clash: organisationData.constraints?.hard_constraints?.no_section_clash || true,
        break_periods_fixed: organisationData.constraints?.hard_constraints?.break_periods_fixed || [],
        lunch_period_fixed: organisationData.constraints?.hard_constraints?.lunch_period_fixed || null,
        mentorship_period_fixed: organisationData.constraints?.hard_constraints?.mentorship_period_fixed || null,
        max_classes_per_day_per_section: organisationData.constraints?.hard_constraints?.max_classes_per_day_per_section || 7,
        min_classes_per_week_per_subject: organisationData.constraints?.hard_constraints?.min_classes_per_week_per_subject || true,
        lab_duration_consecutive: organisationData.constraints?.hard_constraints?.lab_duration_consecutive || true,
        faculty_availability: organisationData.constraints?.hard_constraints?.faculty_availability || true,
        section_room_assignment: organisationData.constraints?.hard_constraints?.section_room_assignment || true
      },
      soft_constraints: {
        balanced_daily_load: {
          weight: organisationData.constraints?.soft_constraints?.balanced_daily_load?.weight || 0.3,
          max_deviation: organisationData.constraints?.soft_constraints?.balanced_daily_load?.max_deviation || 2
        },
        faculty_preference_slots: {
          weight: organisationData.constraints?.soft_constraints?.faculty_preference_slots?.weight || 0.2
        },
        minimize_faculty_travel: {
          weight: organisationData.constraints?.soft_constraints?.minimize_faculty_travel?.weight || 0.15
        },
        morning_heavy_subjects: {
          weight: organisationData.constraints?.soft_constraints?.morning_heavy_subjects?.weight || 0.1,
          subjects: organisationData.constraints?.soft_constraints?.morning_heavy_subjects?.subjects || []
        },
        avoid_single_period_gaps: {
          weight: organisationData.constraints?.soft_constraints?.avoid_single_period_gaps?.weight || 0.15
        },
        distribute_subjects_evenly: {
          weight: organisationData.constraints?.soft_constraints?.distribute_subjects_evenly?.weight || 0.1
        },
        minimize_free_periods: {
          weight: organisationData.constraints?.soft_constraints?.minimize_free_periods?.weight || 0.25
        }
      }
    },
    special_requirements: {
      mentorship_break: {
        period: organisationData.special_requirements?.mentorship_break?.period || null,
        duration: organisationData.special_requirements?.mentorship_break?.duration || 1,
        all_sections: organisationData.special_requirements?.mentorship_break?.all_sections || true
      },
      library_periods: {
        sections: organisationData.special_requirements?.library_periods?.sections || [],
        periods_per_week: organisationData.special_requirements?.library_periods?.periods_per_week || 1,
        flexible: organisationData.special_requirements?.library_periods?.flexible || true
      },
      project_work: {
        sections: organisationData.special_requirements?.project_work?.sections || [],
        periods_per_week: organisationData.special_requirements?.project_work?.periods_per_week || 8,
        flexible_scheduling: organisationData.special_requirements?.project_work?.flexible_scheduling || true
      },
      tutorial_classes: {
        subjects: organisationData.special_requirements?.tutorial_classes?.subjects || [],
        marked_as: organisationData.special_requirements?.tutorial_classes?.marked_as || "T",
        duration: organisationData.special_requirements?.tutorial_classes?.duration || 1
      },
      open_electives: {
        cross_department: organisationData.special_requirements?.open_electives?.cross_department || true,
        faculty_rotation: organisationData.special_requirements?.open_electives?.faculty_rotation || true
      },
      minors_honors: {
        delivery_mode: organisationData.special_requirements?.minors_honors?.delivery_mode || "online",
        platform: organisationData.special_requirements?.minors_honors?.platform || "Google Meet",
        sections: organisationData.special_requirements?.minors_honors?.sections || [],
        periods: organisationData.special_requirements?.minors_honors?.periods || []
      },
      lab_batch_division: {
        max_students_per_batch: organisationData.special_requirements?.lab_batch_division?.max_students_per_batch || 15,
        batch_naming: organisationData.special_requirements?.lab_batch_division?.batch_naming || ["A", "B", "C", "D"],
        rotation_labs: organisationData.special_requirements?.lab_batch_division?.rotation_labs || []
      }
    },
    genetic_algorithm_params: {
      population_size: organisationData.genetic_algorithm_params?.population_size || 50,
      generations: organisationData.genetic_algorithm_params?.generations || 200,
      mutation_rate: organisationData.genetic_algorithm_params?.mutation_rate || 0.2,
      crossover_rate: organisationData.genetic_algorithm_params?.crossover_rate || 0.8,
      elite_size: organisationData.genetic_algorithm_params?.elite_size || 5,
      tournament_size: organisationData.genetic_algorithm_params?.tournament_size || 3,
      early_stopping_patience: organisationData.genetic_algorithm_params?.early_stopping_patience || 5,
      fitness_weights: {
        coverage_weight: organisationData.genetic_algorithm_params?.fitness_weights?.coverage_weight || 1000.0,
        balanced_daily_load_weight: organisationData.genetic_algorithm_params?.fitness_weights?.balanced_daily_load_weight || 50.0,
        faculty_pref_weight: organisationData.genetic_algorithm_params?.fitness_weights?.faculty_pref_weight || 10.0,
        consecutive_periods_weight: organisationData.genetic_algorithm_params?.fitness_weights?.consecutive_periods_weight || 30.0,
        faculty_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.faculty_clash_penalty || 2000.0,
        room_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.room_clash_penalty || 2000.0,
        section_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.section_clash_penalty || 2000.0,
        min_classes_violation_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.min_classes_violation_penalty || 500.0,
        gap_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.gap_penalty || 100.0,
        free_period_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.free_period_penalty || 25.0,
        room_assignment_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.room_assignment_penalty || 1000.0,
        placement_failed_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.placement_failed_penalty || 300.0
      }
    }
  };

  const response = await axios.post(`${FLASK_URL}/api/generate`, transformedData, {
    withCredentials: true
  });
  if (!response || !response.data) throw new ApiError(500, "Failed to start generation");

  console.log("Here is the response", response);
  const apiResponse = response.data.data.faculty;
  console.log(`Fetched ${Object.keys(apiResponse).length} faculty timetables from API`);


  const facultyIds = Object.keys(apiResponse);

  for (const key of facultyIds) {
    const facultyData = apiResponse[key];



    const generateFacultyId = () => {
      return `F${nanoid(5).toUpperCase()}`;
    };
    const dbFacultyId = generateFacultyId();

    await FacultyTimetable.findOneAndUpdate(
      {
        faculty_id: dbFacultyId,
        organisationId,
        course: course.trim().toLowerCase(),
        year: year.trim().toLowerCase(),
        semester: semester.trim().toLowerCase()
      },
      {
        ...facultyData,
        faculty_id: dbFacultyId,
        organisationId
      },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );
  }

  const sectionsObj = response.data.data.sections
  const sectionsArr = Object.values(sectionsObj);

  try {

const ops = sectionsArr.map(sec => ({
  updateOne: {
    filter: {
      section_id: sec.section_id,
      organisationId,
      course: course.trim().toLowerCase(),
      year: year.trim().toLowerCase(),
      semester: semester.trim().toLowerCase()
    },
    update: {
      $set: {
        organisationId,
        course: course.trim().toLowerCase(),
        year: year.trim().toLowerCase(),
        semester: semester.trim().toLowerCase(),
        section_id: sec.section_id,
        section_name: sec.section_name,
        specialization: sec.specialization || "",
        periods: sec.periods || {},
        timetable: sec.timetable || {}
      }
    },
    upsert: true
  }
}));


    await SectionTimetable.bulkWrite(ops, { ordered: false });
  }
  catch (err) {
    console.log("Unable to save the section timetable", err);
    throw new ApiError(500, "Error while saving data! Try again later")
  }




  return res.json(new ApiResponse(200, transformedData, "Timetable generated and saved successfully"));
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
  const organisationId = req.organisation?._id
  let docs = await FacultyTimetable.find({ organisationId }).lean();

  if (!docs || docs.length === 0) {
    throw new ApiError(400, "No timetables found");
  }
  const formattedData = {};
  docs.forEach(doc => {
    const { _id, __v, createdAt, updatedAt, organisationId: orgId, ...cleanDoc } = doc;
    formattedData[doc.faculty_id] = cleanDoc;
  });
  return res.status(200).json(
    new ApiResponse(200, formattedData, "Faculty timetables fetched successfully")
  );
}

);

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


export const getFacultyTimetable = (req, res) => {
  console.log("as");
}