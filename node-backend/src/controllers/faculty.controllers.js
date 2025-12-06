import { Faculty } from "../models/faculty.model.js";
import { Organisation } from "../models/organisation.model.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { Timetable } from "../models/timetable.model.js";
import { TimetableRequest } from "../models/timetableRequest.model.js"; 
import crypto from "crypto";
import OrganisationData from "./organisationData.controllers.js"
import {sendEmail} from "../utils/sendMail.js"; // utility function to send email
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import ApiResponse from "../utils/apiResponse.js";
/**
 * @desc Organisation Admin adds a new Faculty
 * @route POST /api/faculty/create
 * @access Private (Organisation Admin only)
 */
export const createFaculty = async (req, res) => {
  try {
    const organisationId  = req.organisation?._id; // comes from JWT of logged-in org admin
  
    console.log("Organisation Id",organisationId)

    const { facultyName, email, contactNumber, departmentId, subjectsTaught, coursesIn, password } = req.body;

    // check if org exists
    const org = await Organisation.findById(organisationId);
    if (!org) return res.status(404).json({ message: "Organisation not found" });

    // check if faculty email already exists
    const existingFaculty = await Faculty.findOne({ email });
    if (existingFaculty) {
      return res.status(400).json({ message: "Faculty with this email already exists" });
    }

    // hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    const faculty = new Faculty({
      organisationId,
      facultyName,
      email,
      contactNumber,
      departmentId,
      subjectsTaught,
      coursesIn,
      password: hashedPassword,
    });

    await faculty.save();

    res.status(201).json({
      message: "Faculty created successfully",
      faculty: {
        id: faculty._id,
        facultyName: faculty.facultyName,
        email: faculty.email,
        departmentId: faculty.departmentId,
      },
    });
  } catch (err) {
    res.status(500).json({ message: "Error creating faculty", error: err.message });
  }
};


export const deleteFaculty = asyncHandler(async(req,res)=>{
    console.log("I am hitted  delte fac")   
  const organisationId = req.organisation._id;
 console.log("fgc u rtf ",organisationId)


 const facId = req.params.facId;

 console.log("fac Id is here",facId)


 if(!facId)
 {
  throw new ApiError(400,"Sorry the id not found")
 }
  const deletedFaculty = await Faculty.findByIdAndDelete(facId)


if(!deleteFaculty)
{
  throw new ApiError(400,"Something went wrong")
}

return res.status(200).json(
  new ApiResponse(200,deleteFaculty,"deleted")
)

})

export const getAllFaculty = asyncHandler(async(req,res)=>{

const organisationId = req.organisation._id;


const faculties = await Faculty.find({organisationId});

if(!faculties)
{
  throw new ApiError(400,"No faculty has been found");
}

return res.status(202)
.json(
  new ApiResponse(200,faculties,"Faculties fetched successfully")
)

})


/**
 * @desc Faculty Login
 * @route POST /api/faculty/login
 * @access Public
 */
export const loginFaculty = async (req, res) => {
  try {
    const { email, password } = req.body;

    const faculty = await Faculty.findOne({ email });
    if (!faculty) return res.status(400).json({ message: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, faculty.password);
    if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

    if (faculty.status !== "active") {
      return res.status(403).json({ message: "Your account is not active. Contact admin." });
    }

    // generate token
    const token = jwt.sign(
      { facultyId: faculty._id, organisationId: faculty.organisationId, role: "faculty" },
      process.env.JWT_SECRET,
      { expiresIn: "1d" }
    );

    res.json({
      message: "Login successful",
      token,
      faculty: {
        id: faculty._id,
        facultyName: faculty.facultyName,
        email: faculty.email,
        organisationId: faculty.organisationId,
      },
    });
  } catch (err) {
    res.status(500).json({ message: "Error logging in", error: err.message });
  }
};

/**
 * @desc Get Faculty Profile
 * @route GET /api/faculty/profile
 * @access Private (Faculty)
 */
export const getFacultyProfile = async (req, res) => {
  try {
    const { facultyId } = req.user; // from JWT
    const faculty = await Faculty.findById(facultyId)
      .populate("organisationId", "organisationName")
      .populate("departmentId", "departmentName")
      .populate("subjectsTaught", "subjectName")
      .populate("coursesIn", "courseName");

    if (!faculty) return res.status(404).json({ message: "Faculty not found" });

    res.json(faculty);
  } catch (err) {
    res.status(500).json({ message: "Error fetching profile", error: err.message });
  }
};


/**
 * @desc Get Faculty Timetable
 * @route GET /api/faculty/timetable
 * @access Private (Faculty)
 */
export const getFacultyTimetable = async (req, res) => {
  try {
    const { facultyId } = req.user;

    const timetable = await Timetable.find({ facultyId })
      .populate("courseId", "courseName")
      .populate("subjectId", "subjectName")
      .populate("departmentId", "departmentName");

    if (!timetable || timetable.length === 0) {
      return res.status(404).json({ message: "No timetable found" });
    }

    res.json(timetable);
  } catch (err) {
    res.status(500).json({ message: "Error fetching timetable", error: err.message });
  }
};

/**
 * @desc Faculty Requests Timetable Update
 * @route POST /api/faculty/request-update
 * @access Private (Faculty)
 */
export const requestTimetableUpdate = async (req, res) => {
  try {
    const { facultyId, organisationId } = req.user;
    const { message, preferredSlots } = req.body;

    if (!message) {
      return res.status(400).json({ message: "Request message is required" });
    }

    const request = new TimetableRequest({
      facultyId,
      organisationId,
      message,
      preferredSlots, // optional: array of {day, timeSlot}
      status: "pending",
    });

    await request.save();

    res.status(201).json({
      message: "Timetable update request submitted successfully",
      request,
    });
  } catch (err) {
    res.status(500).json({ message: "Error submitting request", error: err.message });
  }
};


/**
 * @desc Forgot Password - send reset link
 * @route POST /api/faculty/forgot-password
 * @access Public
 */
export const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;
    const faculty = await Faculty.findOne({ email });

    if (!faculty) return res.status(404).json({ message: "No faculty found with this email" });

    // generate reset token
    const resetToken = crypto.randomBytes(32).toString("hex");
    const hashedToken = crypto.createHash("sha256").update(resetToken).digest("hex");

    faculty.resetPasswordToken = hashedToken;
    faculty.resetPasswordExpires = Date.now() + 15 * 60 * 1000; // 15 min expiry
    await faculty.save();

    const resetUrl = `${process.env.CLIENT_URL}/reset-password/${resetToken}`;
    const message = `You requested a password reset. Click here: ${resetUrl}`;

    await sendEmail({ to: faculty.email, subject: "Password Reset", text: message });

    res.json({ message: "Password reset email sent" });
  } catch (err) {
    res.status(500).json({ message: "Error sending reset email", error: err.message });
  }
};

/**
 * @desc Reset Password using token
 * @route POST /api/faculty/reset-password/:token
 * @access Public
 */
export const resetPassword = async (req, res) => {
  try {
    const { token } = req.params;
    const { newPassword } = req.body;

    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
    const faculty = await Faculty.findOne({
      resetPasswordToken: hashedToken,
      resetPasswordExpires: { $gt: Date.now() },
    });

    if (!faculty) return res.status(400).json({ message: "Invalid or expired token" });

    faculty.password = await bcrypt.hash(newPassword, 12);
    faculty.resetPasswordToken = undefined;
    faculty.resetPasswordExpires = undefined;
    await faculty.save();

    res.json({ message: "Password reset successful" });
  } catch (err) {
    res.status(500).json({ message: "Error resetting password", error: err.message });
  }
};
export const getFacultyForCourse = asyncHandler(async (req, res) => {
  const { organisationId } = req.params;
  const { course, year, semester } = req.query;

  const data = await OrganisationData.findOne({ organisationId });
  if (!data) throw new ApiError(404, "Organisation data not found");

  const matchedFaculty = data.faculty.filter(f => {
    if (!f.subjects) return false;

    return f.subjects.some(sub =>
      sub.course === course &&
      sub.year === year &&
      sub.semester === semester
    );
  });

  return res
    .status(200)
    .json(new ApiResponse(200, matchedFaculty, "Faculty fetched successfully"));
});

