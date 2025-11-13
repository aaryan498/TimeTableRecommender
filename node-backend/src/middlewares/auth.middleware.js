import jwt from "jsonwebtoken";
import asyncHandler from "../utils/asyncHandler.js";
import ApiError from "../utils/apiError.js";
import { Organisation } from "../models/organisation.model.js";


const verifyJWT = asyncHandler(async (req, res, next) => {
  let token = null;
   console.log("verifyJWT has been hitted")
  if (req.cookies?.accessToken) { 
    token = req.cookies.accessToken;
  }
  else if (req.headers.authorization?.startsWith("Bearer ")) {
    token = req.headers.authorization.split(" ")[1];
  }
  
  if (!token) {
    throw new ApiError(401, "Access token missing or invalid");
  }

  // Verify token
  let decoded;
  try {
    decoded = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET);
  } catch (err) {
    throw new ApiError(401, "Access token is invalid or expired");
  }

 
  const organisation = await Organisation.findById(decoded?._id).select("-password -refreshToken");
  if (!Organisation) {
    throw new ApiError(404, "Organisation not found for this token");
  }

  req.organisation = organisation;
  next();
});

export { verifyJWT };
 export const options = {
  httpOnly:true,
  secure:true,
  sameSite: "none",
 maxAge: 7 * 24 * 60 * 60 * 1000 
 }

