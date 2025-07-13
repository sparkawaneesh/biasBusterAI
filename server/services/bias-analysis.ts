import { AnalysisRequest } from "@shared/schema";
import { urlScraper, ScrapedContent } from "./url-scraper";

const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.GROQ_API_KEY_ENV_VAR || "";

interface BiasAnalysisResult {
  overallScore: number;
  genderScore: number;
  racialScore: number;
  politicalScore: number;
  culturalScore: number;
  detailedReport: any;
  wordCount: number;
  scrapedContent?: ScrapedContent;
}

export class GroqBiasAnalyzer {
  private apiKey: string;
  private baseUrl: string = "https://api.groq.com/openai/v1";

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async analyzeBias(request: AnalysisRequest): Promise<BiasAnalysisResult> {
    let contentToAnalyze = request.content;
    let wordCount = request.content.split(/\s+/).length;
    let scrapedContent: ScrapedContent | undefined;

    // If analyzing a URL, scrape the content first
    if (request.inputType === 'url' && request.url) {
      try {
        scrapedContent = await urlScraper.scrapeArticle(request.url);
        contentToAnalyze = scrapedContent.content;
        wordCount = scrapedContent.wordCount;
      } catch (error) {
        console.error('Error scraping URL:', error);
        throw new Error(`Failed to scrape content from URL: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
    
    const prompt = this.buildPrompt({ ...request, content: contentToAnalyze });
    
    try {
      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama3-8b-8192',
          messages: [
            {
              role: 'system',
              content: 'You are an expert bias detection AI that analyzes text for various forms of bias including gender, racial, political, and cultural bias. Provide detailed, actionable feedback.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.1,
          max_tokens: 2000,
        }),
      });

      if (!response.ok) {
        throw new Error(`GROQ API error: ${response.status}`);
      }

      const data = await response.json();
      const analysisResult = this.parseAnalysisResult(data.choices[0].message.content, request);
      
      return {
        ...analysisResult,
        wordCount,
        scrapedContent,
      };
    } catch (error) {
      console.error('Error calling GROQ API:', error);
      throw new Error('Failed to analyze bias with GROQ API');
    }
  }

  private buildPrompt(request: AnalysisRequest): string {
    const focusArea = request.analysisType === 'comprehensive' ? 
      'all types of bias (gender, racial, political, cultural)' : 
      `${request.analysisType} bias specifically`;

    return `
Analyze the following text for ${focusArea} with ${request.sensitivity} sensitivity level.

Text to analyze:
"${request.content}"

Please provide your analysis in the following JSON format:
{
  "overallScore": [0-10 score],
  "genderScore": [0-10 score],
  "racialScore": [0-10 score], 
  "politicalScore": [0-10 score],
  "culturalScore": [0-10 score],
  "sections": [
    {
      "type": "gender|racial|political|cultural",
      "riskLevel": "low|medium|high",
      "issues": [
        {
          "description": "Description of the issue",
          "location": "Specific text location or line reference",
          "severity": "low|medium|high"
        }
      ],
      "recommendations": [
        "Specific recommendation for improvement"
      ],
      "suggestedRevision": "Improved version of problematic text"
    }
  ],
  "summary": {
    "priorityActions": ["Action 1", "Action 2"],
    "strengths": ["Strength 1", "Strength 2"],
    "impact": "Description of expected impact from improvements"
  }
}

Scoring guidelines:
- 0-3: Low bias/risk
- 4-6: Medium bias/risk  
- 7-10: High bias/risk

Focus on constructive feedback and actionable recommendations.
`;
  }

  private parseAnalysisResult(content: string, request: AnalysisRequest): Omit<BiasAnalysisResult, 'wordCount'> {
    try {
      // Try to extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const result = JSON.parse(jsonMatch[0]);
        return {
          overallScore: result.overallScore || 0,
          genderScore: result.genderScore || 0,
          racialScore: result.racialScore || 0,
          politicalScore: result.politicalScore || 0,
          culturalScore: result.culturalScore || 0,
          detailedReport: result,
        };
      }
    } catch (error) {
      console.error('Error parsing GROQ response:', error);
    }

    // Fallback if parsing fails
    return {
      overallScore: 5,
      genderScore: 5,
      racialScore: 5,
      politicalScore: 5,
      culturalScore: 5,
      detailedReport: {
        sections: [],
        summary: {
          priorityActions: ["Unable to generate detailed analysis"],
          strengths: ["Analysis incomplete"],
          impact: "Please try again with different content"
        }
      },
    };
  }
}

export const biasAnalyzer = new GroqBiasAnalyzer(GROQ_API_KEY);
