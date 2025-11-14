import mongoose, { Schema } from "mongoose";
import mongoosePaginate from "mongoose-paginate-v2";
import mongooseAggregatePaginate from "mongoose-aggregate-paginate-v2";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

const organisationSchema = new Schema(
  {
    organisationName: {
      type: String,
      trim: true,
      required: true,
      lowercase: true,
      maxLength:100
    },
    organisationEmail: {
      type: String,
      unique: true,
      index: true,
      lowercase: true,
      required: true,
      maxLength:50
    },
    isEmailVerified:{
      type:Boolean,
      required:true,
      default:false
    },
    organisationContactNumber: {
      type: String,
      unique: true,
      required: true,
      maxLength:10
    },
    password: {
      type: String,
      trim: true,
      required: true,
      select:false
    },
    avatar: {
      type: String,
    },
  },
  { timestamps: true }
);

organisationSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

// Compare passwords
organisationSchema.methods.isPasswordCorrect = async function (password) {
  return bcrypt.compare(password, this.password);
};

// JWT: Access Token
organisationSchema.methods.generateAccessToken = function () {
  return jwt.sign(
    {
      _id: this._id,
      organisationEmail: this.organisationEmail,
      organisationName: this.organisationName,
    },
    process.env.ACCESS_TOKEN_SECRET,
    {
      expiresIn: process.env.ACCESS_TOKEN_EXPIRY,
    }
  );
};

// JWT: Refresh Token
organisationSchema.methods.generateRefreshToken = function () {
  return jwt.sign(
    { _id: this._id },
    process.env.REFRESH_TOKEN_SECRET,
    {
      expiresIn: process.env.REFRESH_TOKEN_EXPIRY,
    }
  );
};

// Plugins
organisationSchema.plugin(mongoosePaginate);
organisationSchema.plugin(mongooseAggregatePaginate);

// ✅ Prevent OverwriteModelError
export const Organisation =
  mongoose.models.Organisation ||
  mongoose.model("Organisation", organisationSchema);
