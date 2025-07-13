import { pgTable, text, serial, integer, boolean, timestamp, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const biasAnalyses = pgTable("bias_analyses", {
  id: serial("id").primaryKey(),
  content: text("content").notNull(),
  url: text("url"),
  analysisType: text("analysis_type").notNull(),
  sensitivity: text("sensitivity").notNull(),
  inputType: text("input_type").notNull().default("text"),
  overallScore: integer("overall_score").notNull(),
  genderScore: integer("gender_score").notNull(),
  racialScore: integer("racial_score").notNull(),
  politicalScore: integer("political_score").notNull(),
  culturalScore: integer("cultural_score").notNull(),
  detailedReport: json("detailed_report").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  wordCount: integer("word_count").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertBiasAnalysisSchema = createInsertSchema(biasAnalyses).omit({
  id: true,
  createdAt: true,
});

export const analysisRequestSchema = z.object({
  content: z.string().min(1).max(5000),
  url: z.string().url().optional(),
  analysisType: z.enum(["comprehensive", "gender", "racial", "political", "cultural"]),
  sensitivity: z.enum(["low", "standard", "high"]),
  inputType: z.enum(["text", "url"]).default("text"),
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertBiasAnalysis = z.infer<typeof insertBiasAnalysisSchema>;
export type BiasAnalysis = typeof biasAnalyses.$inferSelect;
export type AnalysisRequest = z.infer<typeof analysisRequestSchema>;
