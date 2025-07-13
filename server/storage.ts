import { users, biasAnalyses, type User, type InsertUser, type BiasAnalysis, type InsertBiasAnalysis } from "@shared/schema";

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  createBiasAnalysis(analysis: InsertBiasAnalysis): Promise<BiasAnalysis>;
  getBiasAnalyses(limit?: number): Promise<BiasAnalysis[]>;
  getBiasAnalysis(id: number): Promise<BiasAnalysis | undefined>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private biasAnalyses: Map<number, BiasAnalysis>;
  private currentUserId: number;
  private currentAnalysisId: number;

  constructor() {
    this.users = new Map();
    this.biasAnalyses = new Map();
    this.currentUserId = 1;
    this.currentAnalysisId = 1;
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createBiasAnalysis(insertAnalysis: InsertBiasAnalysis): Promise<BiasAnalysis> {
    const id = this.currentAnalysisId++;
    const analysis: BiasAnalysis = { 
      ...insertAnalysis, 
      id, 
      createdAt: new Date()
    };
    this.biasAnalyses.set(id, analysis);
    return analysis;
  }

  async getBiasAnalyses(limit: number = 10): Promise<BiasAnalysis[]> {
    return Array.from(this.biasAnalyses.values())
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
      .slice(0, limit);
  }

  async getBiasAnalysis(id: number): Promise<BiasAnalysis | undefined> {
    return this.biasAnalyses.get(id);
  }
}

export const storage = new MemStorage();
