import mongoose, { Schema } from "mongoose";

const GeneratedSolutionSchema = new Schema(
  {
    organisationId: {
      type: Schema.Types.ObjectId,
      ref: "Organisation",
      required: true,
    },
    course: {
      type: String,
      required: true,
      trim: true,
      lowercase: true,
    },
    year: {
      type: String,
      required: true,
      trim: true,
      lowercase: true,
    },
    semester: {
      type: String,
      required: true,
      trim: true,
      lowercase: true,
    },
    rank: {
      type: Number,
      required: true,
    },
    fitness: {
      type: Number,
    },
    constraint_violations: {
      type: Schema.Types.Mixed,
      default: {},
    },
    sections: {
      type: Schema.Types.Mixed,
      default: {},
    },
    faculty: {
      type: Schema.Types.Mixed,
      default: {},
    },
    statistics: {
      type: Schema.Types.Mixed,
      default: {},
    },
    isApproved: {
      type: Boolean,
      default: false,
    },
  },
  { timestamps: true }
);

GeneratedSolutionSchema.index(
  {
    organisationId: 1,
    course: 1,
    year: 1,
    semester: 1,
    rank: 1,
  },
  { unique: true }
);

export const GeneratedSolution = mongoose.model(
  "GeneratedSolution",
  GeneratedSolutionSchema
);
