import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { biasAnalyzer } from "./services/bias-analysis";
import { analysisRequestSchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  app.post("/api/analyze", async (req, res) => {
    try {
      const validatedData = analysisRequestSchema.parse(req.body);
      
      const result = await biasAnalyzer.analyzeBias(validatedData);
      
      const analysis = await storage.createBiasAnalysis({
        content: validatedData.content,
        url: validatedData.url,
        analysisType: validatedData.analysisType,
        sensitivity: validatedData.sensitivity,
        inputType: validatedData.inputType,
        ...result,
      });
      
      res.json(analysis);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid request data", errors: error.errors });
      } else {
        console.error("Analysis error:", error);
        res.status(500).json({ message: "Failed to analyze content" });
      }
    }
  });

  app.get("/api/analyses", async (req, res) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
      const analyses = await storage.getBiasAnalyses(limit);
      res.json(analyses);
    } catch (error) {
      console.error("Error fetching analyses:", error);
      res.status(500).json({ message: "Failed to fetch analysis history" });
    }
  });

  app.get("/api/analyses/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const analysis = await storage.getBiasAnalysis(id);
      
      if (!analysis) {
        return res.status(404).json({ message: "Analysis not found" });
      }
      
      res.json(analysis);
    } catch (error) {
      console.error("Error fetching analysis:", error);
      res.status(500).json({ message: "Failed to fetch analysis" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
