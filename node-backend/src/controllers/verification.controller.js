import asyncHandler from "../utils/asyncHandler.js";
import { redisClient } from "../middlewares/otp.middleware.js";
import ApiError from "../utils/apiError.js";
import ApiResponse from "../utils/apiResponse.js";
import bcrypt from "bcrypt";
import { sendEmail } from "../utils/sendMail.js";
import { Organisation } from "../models/organisation.model.js";
import jwt from "jsonwebtoken"

const OTP_EXPIRY = 2 * 60; 
const RATE_LIMIT = 10;    
const RESEND_LIMIT = 60;   

export const generateOTP = () =>
  Math.floor(100000 + Math.random() * 900000).toString();


export const verifyOtp = async (organisationEmail, otp, purpose) => {
  const key = `otp:data:${purpose}:${organisationEmail}`;

  const hashedOtp = await redisClient.get(key);

  if (!hashedOtp) return false;

  return await bcrypt.compare(otp, hashedOtp);
};


export const sendOTP = asyncHandler(async (req, res) => {
  const organisationEmail = req.body?.organisationEmail || req.user?.organisationEmail;
  const { purpose } = req.params;
  let isEmailVerified = true
   if(!purpose)
   {
    throw new ApiError(400,"Purpose is not defined")
   }
  if (!organisationEmail) {
    throw new ApiError(400, "organisationEmail is required");
  }
 
  if(purpose ==="register")
  {
      isEmailVerified = false;
  }
  
  const isOrganisationExist  = await Organisation.findOne({
    organisationEmail,
    isEmailVerified
  })
 
  

  if(!isOrganisationExist)
  {
    throw new ApiError(400,"Organisation does not exist! Please Register  first")
  }
  if (!purpose) {
    throw new ApiError(400, "OTP purpose is required");
  }

  if (!/^\S+@\S+\.\S+$/.test(organisationEmail)) {
    throw new ApiError(400, "Please provide a valid organisationEmail");
  }

  const now = Date.now();


  const lastSentKey = `otp:lastSent:${purpose}:${organisationEmail}`;
  const lastSent = await redisClient.get(lastSentKey);

  if (lastSent && now - parseInt(lastSent) < RESEND_LIMIT * 1000) {
    throw new ApiError(
      429,
      `Please wait ${RESEND_LIMIT} seconds before requesting another OTP.`
    );
  }

  const otpCountKey = `otp:count:${purpose}:${organisationEmail}`;
  const sentCount = await redisClient.get(otpCountKey);

  if (sentCount && parseInt(sentCount) >= RATE_LIMIT) {
    throw new ApiError(429, "OTP limit exceeded. Try again after 1 hour.");
  }


  const otp = generateOTP();
  const hashedOtp = await bcrypt.hash(otp, 10);


  const otpKey = `otp:data:${purpose}:${organisationEmail}`;

  await redisClient.set(otpKey, hashedOtp, { EX: OTP_EXPIRY });


  await redisClient.set(lastSentKey, now.toString(), { EX: RESEND_LIMIT });

  const isNewKey = !sentCount;
  await redisClient.incr(otpCountKey);
  if (isNewKey) await redisClient.expire(otpCountKey, 3600);

   sendEmail(
    organisationEmail,
    "Your OTP for Organisation Verification",
    `Your OTP is: ${otp}. It expires in 5 minutes.`,
    purpose
  );

  return res
    .status(202)
    .json(new ApiResponse(200, {}, "OTP has been sent successfully"));
});


export const checkOtp = asyncHandler(async (req, res) => {
  const { organisationEmail, otp, purpose } = req.body;

  if (!organisationEmail || !otp || !purpose) {
    throw new ApiError(400, "Please provide organisationEmail, OTP and purpose");
  }

  const isCorrect = await verifyOtp(organisationEmail, otp, purpose);
   
  if(!isCorrect)
  {
    throw new ApiError(400,"Otp Incorrect");
  }
 if(purpose==="reset-password")
 {
    
  const organisation = await Organisation.findOne({organisationEmail}).lean()
  
  if(!organisation)
  {
    throw new ApiError(400,"Organisation Not Found! Register first")
  }

  res.cookie("otpToken", jwt.sign(
      {
        _id: this._id,
        organisationEmail: organisation.organisationEmail,
        organisationName: organisation.organisationName,
      },
      process.env.OTP_TOKEN_SECRET,
      {
        expiresIn: process.env.OTP_TOKEN_EXPIRY,
      }
    ))


 }

  return res
    .status(200)
    .json(new ApiResponse(200, { isCorrect }, "OTP verification result"));
});
