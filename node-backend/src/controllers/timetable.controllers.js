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

import { GeneratedSolution } from "../models/generatedSolution.model.js";

const { FLASK_URL } = process.env;



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
// export const startTimeTableCreation = asyncHandler(async (req, res) => {
//   const organisationId = req.organisation?._id;
//   const { course, year, semester } = req.query;

//   if (!organisationId) {
//     throw new ApiError(400, "Login First");
//   }

//   if (!course?.trim() || !year?.trim() || !semester?.trim()) {
//     throw new ApiError(400, "Course, Year and Semester are required");
//   }

//   const c = course.trim().toLowerCase();
//   const y = year.trim().toLowerCase();
//   const s = semester.trim().toLowerCase();

//   const organisationData = await OrganisationData.findOne({
//     organisationId,
//     course: c,
//     year: y,
//     semester: s,
//   });

//   if (!organisationData) {
//     return res.status(404).json({
//       success: false,
//       statusCode: 404,
//       message: "Organisation data not found",
//       data: null,
//     });
//   }

// const transformedData = {
//     college_info: {
//       name: organisationData.college_info?.name || "",
//       session: organisationData.college_info?.session || "",
//       effective_date: organisationData.college_info?.effective_date
//         ? new Date(organisationData.college_info.effective_date).toISOString().split('T')[0]
//         : ""
//     },
//     time_slots: {
//       periods: (organisationData.time_slots?.periods || []).map(period => ({
//         id: period.id,
//         start_time: period.start_time,
//         end_time: period.end_time
//       })),
//       working_days: organisationData.time_slots?.working_days || [],
//       break_periods: organisationData.time_slots?.break_periods || [],
//       lunch_period: organisationData.time_slots?.lunch_period || null,
//       mentorship_period: organisationData.time_slots?.mentorship_period || null
//     },
//     departments: (organisationData.departments || []).map(dept => ({
//       dept_id: dept.dept_id,
//       name: dept.name,
//       sections: (dept.sections || []).map(section => ({
//         section_id: section.section_id,
//         name: section.name,
//         semester: typeof section.semester === 'string' ? parseInt(section.semester) || section.semester : section.semester,
//         year: typeof section.year === 'string' ? parseInt(section.year) || section.year : section.year,
//         room: section.room,
//         student_count: section.student_count,
//         coordinator: section.coordinator,
//         specialization: section.specialization || ""
//       }))
//     })),
//     subjects: (organisationData.subjects || []).map(subject => ({
//       subject_id: subject.subject_id,
//       name: subject.name,
//       type: subject.type,
//       credits: subject.credits,
//       lectures_per_week: subject.lectures_per_week,
//       semester: subject.semester,
//       departments: subject.departments || [],
//       min_classes_per_week: subject.min_classes_per_week,
//       max_classes_per_day: subject.max_classes_per_day,
//       tutorial_sessions: subject.tutorial_sessions || 0,
//       specialization: subject.specialization || "",
//       flexible_timing: subject.flexible_timing || false
//     })),
//     labs: (organisationData.labs || []).map(lab => ({
//       lab_id: lab.lab_id,
//       name: lab.name,
//       type: lab.type,
//       credits: lab.credits,
//       sessions_per_week: lab.sessions_per_week,
//       duration_hours: lab.duration_hours,
//       semester: lab.semester,
//       departments: lab.departments || [],
//       lab_rooms: lab.lab_rooms || [],
//       specialization: lab.specialization || ""
//     })),
//     faculty: (organisationData.faculty || []).map(faculty => ({
//       faculty_id: faculty.faculty_id,
//       name: faculty.name,
//       department: faculty.department,
//       designation: faculty.designation,
//       subjects: faculty.subjects || [],
//       max_hours_per_week: faculty.max_hours_per_week,
//       avg_leaves_per_month: faculty.avg_leaves_per_month,
//       preferred_time_slots: faculty.preferred_time_slots || []
//     })),
//     rooms: (organisationData.rooms || []).map(room => ({
//       room_id: room.room_id,
//       name: room.name,
//       type: room.type,
//       capacity: room.capacity,
//       department: room.department || "",
//       equipment: room.equipment || []
//     })),
//     constraints: {
//       hard_constraints: {
//         no_faculty_clash: organisationData.constraints?.hard_constraints?.no_faculty_clash || true,
//         no_room_clash: organisationData.constraints?.hard_constraints?.no_room_clash || true,
//         no_section_clash: organisationData.constraints?.hard_constraints?.no_section_clash || true,
//         break_periods_fixed: organisationData.constraints?.hard_constraints?.break_periods_fixed || [],
//         lunch_period_fixed: organisationData.constraints?.hard_constraints?.lunch_period_fixed || null,
//         mentorship_period_fixed: organisationData.constraints?.hard_constraints?.mentorship_period_fixed || null,
//         max_classes_per_day_per_section: organisationData.constraints?.hard_constraints?.max_classes_per_day_per_section || 7,
//         min_classes_per_week_per_subject: organisationData.constraints?.hard_constraints?.min_classes_per_week_per_subject || true,
//         lab_duration_consecutive: organisationData.constraints?.hard_constraints?.lab_duration_consecutive || true,
//         faculty_availability: organisationData.constraints?.hard_constraints?.faculty_availability || true,
//         section_room_assignment: organisationData.constraints?.hard_constraints?.section_room_assignment || true
//       },
//       soft_constraints: {
//         balanced_daily_load: {
//           weight: organisationData.constraints?.soft_constraints?.balanced_daily_load?.weight || 0.3,
//           max_deviation: organisationData.constraints?.soft_constraints?.balanced_daily_load?.max_deviation || 2
//         },
//         faculty_preference_slots: {
//           weight: organisationData.constraints?.soft_constraints?.faculty_preference_slots?.weight || 0.2
//         },
//         minimize_faculty_travel: {
//           weight: organisationData.constraints?.soft_constraints?.minimize_faculty_travel?.weight || 0.15
//         },
//         morning_heavy_subjects: {
//           weight: organisationData.constraints?.soft_constraints?.morning_heavy_subjects?.weight || 0.1,
//           subjects: organisationData.constraints?.soft_constraints?.morning_heavy_subjects?.subjects || []
//         },
//         avoid_single_period_gaps: {
//           weight: organisationData.constraints?.soft_constraints?.avoid_single_period_gaps?.weight || 0.15
//         },
//         distribute_subjects_evenly: {
//           weight: organisationData.constraints?.soft_constraints?.distribute_subjects_evenly?.weight || 0.1
//         },
//         minimize_free_periods: {
//           weight: organisationData.constraints?.soft_constraints?.minimize_free_periods?.weight || 0.25
//         }
//       }
//     },
//     special_requirements: {
//       mentorship_break: {
//         period: organisationData.special_requirements?.mentorship_break?.period || null,
//         duration: organisationData.special_requirements?.mentorship_break?.duration || 1,
//         all_sections: organisationData.special_requirements?.mentorship_break?.all_sections || true
//       },
//       library_periods: {
//         sections: organisationData.special_requirements?.library_periods?.sections || [],
//         periods_per_week: organisationData.special_requirements?.library_periods?.periods_per_week || 1,
//         flexible: organisationData.special_requirements?.library_periods?.flexible || true
//       },
//       project_work: {
//         sections: organisationData.special_requirements?.project_work?.sections || [],
//         periods_per_week: organisationData.special_requirements?.project_work?.periods_per_week || 8,
//         flexible_scheduling: organisationData.special_requirements?.project_work?.flexible_scheduling || true
//       },
//       tutorial_classes: {
//         subjects: organisationData.special_requirements?.tutorial_classes?.subjects || [],
//         marked_as: organisationData.special_requirements?.tutorial_classes?.marked_as || "T",
//         duration: organisationData.special_requirements?.tutorial_classes?.duration || 1
//       },
//       open_electives: {
//         cross_department: organisationData.special_requirements?.open_electives?.cross_department || true,
//         faculty_rotation: organisationData.special_requirements?.open_electives?.faculty_rotation || true
//       },
//       minors_honors: {
//         delivery_mode: organisationData.special_requirements?.minors_honors?.delivery_mode || "online",
//         platform: organisationData.special_requirements?.minors_honors?.platform || "Google Meet",
//         sections: organisationData.special_requirements?.minors_honors?.sections || [],
//         periods: organisationData.special_requirements?.minors_honors?.periods || []
//       },
//       lab_batch_division: {
//         max_students_per_batch: organisationData.special_requirements?.lab_batch_division?.max_students_per_batch || 15,
//         batch_naming: organisationData.special_requirements?.lab_batch_division?.batch_naming || ["A", "B", "C", "D"],
//         rotation_labs: organisationData.special_requirements?.lab_batch_division?.rotation_labs || []
//       }
//     },
//     genetic_algorithm_params: {
//       population_size: organisationData.genetic_algorithm_params?.population_size || 50,
//       generations: organisationData.genetic_algorithm_params?.generations || 200,
//       mutation_rate: organisationData.genetic_algorithm_params?.mutation_rate || 0.2,
//       crossover_rate: organisationData.genetic_algorithm_params?.crossover_rate || 0.8,
//       elite_size: organisationData.genetic_algorithm_params?.elite_size || 5,
//       tournament_size: organisationData.genetic_algorithm_params?.tournament_size || 3,
//       early_stopping_patience: organisationData.genetic_algorithm_params?.early_stopping_patience || 5,
//       fitness_weights: {
//         coverage_weight: organisationData.genetic_algorithm_params?.fitness_weights?.coverage_weight || 1000.0,
//         balanced_daily_load_weight: organisationData.genetic_algorithm_params?.fitness_weights?.balanced_daily_load_weight || 50.0,
//         faculty_pref_weight: organisationData.genetic_algorithm_params?.fitness_weights?.faculty_pref_weight || 10.0,
//         consecutive_periods_weight: organisationData.genetic_algorithm_params?.fitness_weights?.consecutive_periods_weight || 30.0,
//         faculty_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.faculty_clash_penalty || 2000.0,
//         room_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.room_clash_penalty || 2000.0,
//         section_clash_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.section_clash_penalty || 2000.0,
//         min_classes_violation_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.min_classes_violation_penalty || 500.0,
//         gap_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.gap_penalty || 100.0,
//         free_period_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.free_period_penalty || 25.0,
//         room_assignment_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.room_assignment_penalty || 1000.0,
//         placement_failed_penalty: organisationData.genetic_algorithm_params?.fitness_weights?.placement_failed_penalty || 300.0
//       }
//     }
//   };



//   console.log("Here is the going to send to model")

//   const response = await axios.post(`${FLASK_URL}/api/generate`, transformedData, {
//     withCredentials: true,
//   });

//   if (!response?.data) {
//     throw new ApiError(500, "Failed to start generation");
//   }

//   const facultyDataObj = response.data.data.faculty;
//   const facultyIds = Object.keys(facultyDataObj);

//   for (const key of facultyIds) {
//     const facultyData = facultyDataObj[key];

//     await FacultyTimetable.findOneAndUpdate(
//       {
//         organisationId,
//         course: c,
//         year: y,
//         semester: s,
//         faculty_id: facultyData.faculty_id, // now uses provided id
//       },
//       {
//         ...facultyData,
//         organisationId,
//         course: c,
//         year: y,
//         semester: s,
//       },
//       { new: true, upsert: true }
//     );
//   }

//   const sectionsObj = response.data.data.sections;
//   const sectionsArr = Object.values(sectionsObj);

//   const ops = sectionsArr.map((sec) => ({
//     updateOne: {
//       filter: {
//         organisationId,
//         course: c,
//         year: y,
//         semester: s,
//         section_id: sec.section_id,
//       },
//       update: {
//         $set: {
//           organisationId,
//           course: c,
//           year: y,
//           semester: s,
//           section_id: sec.section_id,
//           section_name: sec.section_name,
//           specialization: sec.specialization || "",
//           periods: sec.periods || {},
//           timetable: sec.timetable || {},
//         },
//       },
//       upsert: true,
//     },
//   }));

//   await SectionTimetable.bulkWrite(ops, { ordered: false });

// return res.json(
//   new ApiResponse(
//     200,
//     {  transformedData,
//       faculty: response.data.data.faculty,
//       sections: response.data.data.sections
//     },
//     "Timetable generated and saved successfully"
//   )
// );

// });



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
    return res.status(404).json(
      new ApiResponse(
        404,
        null,
        "Organisation data not found for this course/year/semester"
      )
    );
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
      periods: (organisationData.time_slots?.periods || []).map((period) => ({
        id: period.id,
        start_time: period.start_time,
        end_time: period.end_time,
      })),
      working_days: organisationData.time_slots?.working_days || [],
      break_periods: organisationData.time_slots?.break_periods || [],
      lunch_period: organisationData.time_slots?.lunch_period || null,
      mentorship_period:
        organisationData.time_slots?.mentorship_period || null,
    },
    departments: (organisationData.departments || []).map((dept) => ({
      dept_id: dept.dept_id,
      name: dept.name,
      sections: (dept.sections || []).map((section) => ({
        section_id: section.section_id,
        name: section.name,
        semester:
          typeof section.semester === "string"
            ? parseInt(section.semester) || section.semester
            : section.semester,
        year:
          typeof section.year === "string"
            ? parseInt(section.year) || section.year
            : section.year,
        room: section.room,
        student_count: section.student_count,
        coordinator: section.coordinator,
        specialization: section.specialization || "",
      })),
    })),
    subjects: (organisationData.subjects || []).map((subject) => ({
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
      flexible_timing: subject.flexible_timing || false,
    })),
    labs: (organisationData.labs || []).map((lab) => ({
      lab_id: lab.lab_id,
      name: lab.name,
      type: lab.type,
      credits: lab.credits,
      sessions_per_week: lab.sessions_per_week,
      duration_hours: lab.duration_hours,
      semester: lab.semester,
      departments: lab.departments || [],
      lab_rooms: lab.lab_rooms || [],
      specialization: lab.specialization || "",
    })),
    faculty: (organisationData.faculty || []).map((faculty) => ({
      faculty_id: faculty.faculty_id,
      name: faculty.name,
      department: faculty.department,
      designation: faculty.designation,
      subjects: faculty.subjects || [],
      max_hours_per_week: faculty.max_hours_per_week,
      avg_leaves_per_month: faculty.avg_leaves_per_month,
      preferred_time_slots: faculty.preferred_time_slots || [],
    })),
    rooms: (organisationData.rooms || []).map((room) => ({
      room_id: room.room_id,
      name: room.name,
      type: room.type,
      capacity: room.capacity,
      department: room.department || "",
      equipment: room.equipment || [],
    })),
    constraints: {
      hard_constraints: {
        no_faculty_clash:
          organisationData.constraints?.hard_constraints?.no_faculty_clash ??
          true,
        no_room_clash:
          organisationData.constraints?.hard_constraints?.no_room_clash ??
          true,
        no_section_clash:
          organisationData.constraints?.hard_constraints?.no_section_clash ??
          true,
        break_periods_fixed:
          organisationData.constraints?.hard_constraints?.break_periods_fixed ||
          [],
        lunch_period_fixed:
          organisationData.constraints?.hard_constraints?.lunch_period_fixed ||
          null,
        mentorship_period_fixed:
          organisationData.constraints?.hard_constraints
            ?.mentorship_period_fixed || null,
        max_classes_per_day_per_section:
          organisationData.constraints?.hard_constraints
            ?.max_classes_per_day_per_section || 7,
        min_classes_per_week_per_subject:
          organisationData.constraints?.hard_constraints
            ?.min_classes_per_week_per_subject ?? true,
        lab_duration_consecutive:
          organisationData.constraints?.hard_constraints
            ?.lab_duration_consecutive ?? true,
        faculty_availability:
          organisationData.constraints?.hard_constraints
            ?.faculty_availability ?? true,
        section_room_assignment:
          organisationData.constraints?.hard_constraints
            ?.section_room_assignment ?? true,
      },
      soft_constraints: {
        balanced_daily_load: {
          weight:
            organisationData.constraints?.soft_constraints?.balanced_daily_load
              ?.weight || 0.3,
          max_deviation:
            organisationData.constraints?.soft_constraints?.balanced_daily_load
              ?.max_deviation || 2,
        },
        faculty_preference_slots: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.faculty_preference_slots?.weight || 0.2,
        },
        minimize_faculty_travel: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.minimize_faculty_travel?.weight || 0.15,
        },
        morning_heavy_subjects: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.morning_heavy_subjects?.weight || 0.1,
          subjects:
            organisationData.constraints?.soft_constraints
              ?.morning_heavy_subjects?.subjects || [],
        },
        avoid_single_period_gaps: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.avoid_single_period_gaps?.weight || 0.15,
        },
        distribute_subjects_evenly: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.distribute_subjects_evenly?.weight || 0.1,
        },
        minimize_free_periods: {
          weight:
            organisationData.constraints?.soft_constraints
              ?.minimize_free_periods?.weight || 0.25,
        },
      },
    },
    special_requirements: {
      mentorship_break: {
        period:
          organisationData.special_requirements?.mentorship_break?.period ||
          null,
        duration:
          organisationData.special_requirements?.mentorship_break?.duration ||
          1,
        all_sections:
          organisationData.special_requirements?.mentorship_break
            ?.all_sections || true,
      },
      library_periods: {
        sections:
          organisationData.special_requirements?.library_periods?.sections ||
          [],
        periods_per_week:
          organisationData.special_requirements?.library_periods
            ?.periods_per_week || 1,
        flexible:
          organisationData.special_requirements?.library_periods?.flexible ??
          true,
      },
      project_work: {
        sections:
          organisationData.special_requirements?.project_work?.sections || [],
        periods_per_week:
          organisationData.special_requirements?.project_work
            ?.periods_per_week || 8,
        flexible_scheduling:
          organisationData.special_requirements?.project_work
            ?.flexible_scheduling ?? true,
      },
      tutorial_classes: {
        subjects:
          organisationData.special_requirements?.tutorial_classes?.subjects ||
          [],
        marked_as:
          organisationData.special_requirements?.tutorial_classes?.marked_as ||
          "T",
        duration:
          organisationData.special_requirements?.tutorial_classes?.duration ||
          1,
      },
      open_electives: {
        cross_department:
          organisationData.special_requirements?.open_electives
            ?.cross_department ?? true,
        faculty_rotation:
          organisationData.special_requirements?.open_electives
            ?.faculty_rotation ?? true,
      },
      minors_honors: {
        delivery_mode:
          organisationData.special_requirements?.minors_honors?.delivery_mode ||
          "online",
        platform:
          organisationData.special_requirements?.minors_honors?.platform ||
          "Google Meet",
        sections:
          organisationData.special_requirements?.minors_honors?.sections || [],
        periods:
          organisationData.special_requirements?.minors_honors?.periods || [],
      },
      lab_batch_division: {
        max_students_per_batch:
          organisationData.special_requirements?.lab_batch_division
            ?.max_students_per_batch || 15,
        batch_naming:
          organisationData.special_requirements?.lab_batch_division
            ?.batch_naming || ["A", "B", "C", "D"],
        rotation_labs:
          organisationData.special_requirements?.lab_batch_division
            ?.rotation_labs || [],
      },
    },
    genetic_algorithm_params: {
      population_size:
        organisationData.genetic_algorithm_params?.population_size || 50,
      generations: organisationData.genetic_algorithm_params?.generations || 200,
      mutation_rate:
        organisationData.genetic_algorithm_params?.mutation_rate || 0.2,
      crossover_rate:
        organisationData.genetic_algorithm_params?.crossover_rate || 0.8,
      elite_size: organisationData.genetic_algorithm_params?.elite_size || 5,
      tournament_size:
        organisationData.genetic_algorithm_params?.tournament_size || 3,
      early_stopping_patience:
        organisationData.genetic_algorithm_params?.early_stopping_patience || 5,
      fitness_weights: {
        coverage_weight:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.coverage_weight || 1000.0,
        balanced_daily_load_weight:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.balanced_daily_load_weight || 50.0,
        faculty_pref_weight:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.faculty_pref_weight || 10.0,
        consecutive_periods_weight:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.consecutive_periods_weight || 30.0,
        faculty_clash_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.faculty_clash_penalty || 2000.0,
        room_clash_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.room_clash_penalty || 2000.0,
        section_clash_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.section_clash_penalty || 2000.0,
        min_classes_violation_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.min_classes_violation_penalty || 500.0,
        gap_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.gap_penalty || 100.0,
        free_period_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.free_period_penalty || 25.0,
        room_assignment_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.room_assignment_penalty || 1000.0,
        placement_failed_penalty:
          organisationData.genetic_algorithm_params?.fitness_weights
            ?.placement_failed_penalty || 300.0,
      },
    },
  };

  const response = await axios.post(`${FLASK_URL}/api/generate`, transformedData, {
    withCredentials: true,
  });

  const gaData = response?.data;
 
    console.log("Here is this coming from flask ",gaData)
  if (!gaData || !Array.isArray(gaData.solutions) || gaData.solutions.length === 0) {
    throw new ApiError(500, "Model did not return any solutions");
  }

  await GeneratedSolution.deleteMany({
    organisationId,
    course: c,
    year: y,
    semester: s,
  });

  const docsToInsert = gaData.solutions.map((sol) => ({
    organisationId,
    course: c,
    year: y,
    semester: s,
    rank: sol.rank,
    fitness: sol.fitness,
    constraint_violations: sol.constraint_violations || {},
    sections: sol.sections || {},
    faculty: sol.faculty || {},
    statistics: sol.statistics || {},
  }));

  const inserted = await GeneratedSolution.insertMany(docsToInsert, {
    ordered: true,
  });

  const summary = inserted
    .slice()
    .sort((a, b) => a.rank - b.rank)
    .map((doc) => ({
      _id: doc._id,
      rank: doc.rank,
      fitness: doc.fitness,
      statistics: doc.statistics,
      isApproved: doc.isApproved,
    }));

  return res.json(
    new ApiResponse(
      200,
      {
        course: c,
        year: y,
        semester: s,
        solutions: summary,
      },
      "Timetable variants generated and stored successfully"
    )
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
export const getGeneratedSolutions = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { course, year, semester } = req.query;

  if (!organisationId) {
    throw new ApiError(401, "Login first");
  }

  if (!course?.trim() || !year?.trim() || !semester?.trim()) {
    throw new ApiError(400, "Course, year, and semester are required");
  }

  const c = course.trim().toLowerCase();
  const y = year.trim().toLowerCase();
  const s = semester.trim().toLowerCase();

  const solutions = await GeneratedSolution.find({
    organisationId,
    course: c,
    year: y,
    semester: s,
  })
    .sort({ rank: 1, createdAt: -1 })
    .lean();

  if (!solutions.length) {
    throw new ApiError(404, "No timetable variants found");
  }

  const payload = solutions.map((sol) => ({
    _id: sol._id,
    rank: sol.rank,
    fitness: sol.fitness,
    statistics: sol.statistics,
    isApproved: sol.isApproved,
  }));

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        course: c,
        year: y,
        semester: s,
        solutions: payload,
      },
      "Generated timetable variants fetched"
    )
  );
});
export const getGeneratedSolutionById = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { id } = req.params;

  if (!organisationId) {
    throw new ApiError(401, "Login first");
  }

  const sol = await GeneratedSolution.findOne({
    _id: id,
    organisationId,
  }).lean();

  if (!sol) {
    throw new ApiError(404, "Solution not found");
  }

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        _id: sol._id,
        course: sol.course,
        year: sol.year,
        semester: sol.semester,
        rank: sol.rank,
        fitness: sol.fitness,
        statistics: sol.statistics,
        sections: sol.sections,
        faculty: sol.faculty,
        isApproved: sol.isApproved,
      },
      "Solution fetched successfully"
    )
  );
});
export const approveGeneratedSolution = asyncHandler(async (req, res) => {
  const organisationId = req.organisation?._id;
  const { solutionId } = req.body;

  if (!organisationId) throw new ApiError(401, "Login first");
  if (!solutionId) throw new ApiError(400, "solutionId is required");

  const sol = await GeneratedSolution.findOne({
    _id: solutionId,
    organisationId,
  }).lean();

  if (!sol) throw new ApiError(404, "Solution not found");

  const { course, year, semester, sections, faculty } = sol;

  // Delete previous approved timetables for this course/year/sem
  await Promise.all([
    SectionTimetable.deleteMany({ organisationId, course, year, semester }),
    FacultyTimetable.deleteMany({ organisationId, course, year, semester }),
  ]);

  //
  // ------------------------
  // SAVE SECTION TIMETABLES
  // ------------------------
  //
  const sectionArr = Object.values(sections || {});

  if (sectionArr.length > 0) {
    const sectionOps = sectionArr.map(sec => ({
      updateOne: {
        filter: {
          organisationId,
          course,
          year,
          semester,
          section_id: sec.section_id
        },
        update: {
          $set: {
            organisationId,
            course,
            year,
            semester,
            section_id: sec.section_id,
            section_name: sec.section_name || "",
            specialization: sec.specialization || "",
            periods: sec.periods || {},
            timetable: sec.timetable || {}
          }
        },
        upsert: true
      }
    }));

    await SectionTimetable.bulkWrite(sectionOps, { ordered: false });
  }

  //
  // ------------------------
  // SAVE FACULTY TIMETABLES
  // ------------------------
  //
  const facultyArr = faculty ? Object.values(faculty) : [];

  if (facultyArr.length > 0) {
    const facultyOps = facultyArr.map(f => ({
      updateOne: {
        filter: {
          organisationId,
          course,
          year,
          semester,
          faculty_id: f.faculty_id
        },
        update: {
          $set: {
            organisationId,
            course,
            year,
            semester,
            faculty_id: f.faculty_id,
            faculty_name: f.faculty_name,           // REQUIRED
            department: f.department,               // REQUIRED
            periods: f.periods || {},               // Map { "1": "08:00-08:50", ... }
            timetable: f.timetable || {}            // Map { "Monday": {...}, "Tuesday": {...} }
          }
        },
        upsert: true
      }
    }));

    await FacultyTimetable.bulkWrite(facultyOps, { ordered: false });
  }

  //
  // ------------------------
  // MARK APPROVED SOLUTION
  // ------------------------
  //

  await GeneratedSolution.updateMany(
    { organisationId, course, year, semester },
    { $set: { isApproved: false } }
  );

  await GeneratedSolution.updateOne(
    { _id: sol._id },
    { $set: { isApproved: true } }
  );

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        course,
        year,
        semester,
        approvedSolution: sol._id
      },
      "Selected timetable variant approved & saved successfully"
    )
  );
});

