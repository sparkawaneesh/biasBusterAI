import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BiasAnalysis } from "@shared/schema";
import { CheckCircle, TrendingUp, BarChart3 } from "lucide-react";

interface ResultsOverviewProps {
  analysis: BiasAnalysis;
}

function getBiasLevel(score: number): { level: string; color: string; bgColor: string } {
  if (score <= 3) return { level: "Low Risk", color: "text-green-800", bgColor: "bg-green-100" };
  if (score <= 6) return { level: "Medium Risk", color: "text-yellow-800", bgColor: "bg-yellow-100" };
  return { level: "High Risk", color: "text-red-800", bgColor: "bg-red-100" };
}

function getProgressColor(score: number): string {
  if (score <= 3) return "bg-green-500";
  if (score <= 6) return "bg-yellow-500";
  return "bg-red-500";
}

export default function ResultsOverview({ analysis }: ResultsOverviewProps) {
  const overallBias = getBiasLevel(analysis.overallScore);

  const biasTypes = [
    { name: "Gender Bias", score: analysis.genderScore },
    { name: "Racial Bias", score: analysis.racialScore },
    { name: "Political Bias", score: analysis.politicalScore },
    { name: "Cultural Bias", score: analysis.culturalScore },
  ];

  return (
    <Card className="bg-gradient-to-br from-white via-blue-50/30 to-purple-50/30 rounded-3xl shadow-xl border border-gray-200/50 mb-8 backdrop-blur-sm animate-in slide-in-from-bottom-4 duration-500 hover:shadow-2xl hover:scale-[1.02] transition-all">
      <CardContent className="p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-3 hover:scale-110 transition-transform duration-300">
              <BarChart3 className="text-white h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                📊 Analysis Results
              </h2>
              <p className="text-sm text-gray-600">Comprehensive bias detection report</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 bg-green-50 rounded-full px-4 py-2 hover:scale-105 transition-transform duration-300">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <span className="text-sm font-medium text-green-700">✅ Completed</span>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 rounded-2xl p-8 mb-8 border border-blue-200/50 hover:shadow-lg transition-all duration-300 hover:scale-[1.01]">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">🎯 Overall Bias Score</h3>
              <p className="text-gray-600">Based on comprehensive analysis across all categories</p>
              <div className="flex items-center space-x-2 mt-3">
                <TrendingUp className="h-4 w-4 text-blue-500" />
                <span className="text-sm text-gray-600">Multi-dimensional analysis</span>
              </div>
            </div>
            <div className="text-right">
              <div className="relative">
                <div className="text-6xl font-bold bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text text-transparent animate-in zoom-in-50 duration-1000">
                  {analysis.overallScore.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500 mt-1">out of 10</div>
              </div>
              <Badge className={`${overallBias.bgColor} ${overallBias.color} mt-3 px-4 py-2 font-semibold animate-in fade-in-50 duration-1000 delay-500`}>
                {overallBias.level}
              </Badge>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {biasTypes.map((type, index) => {
            const bias = getBiasLevel(type.score);
            const progressColor = getProgressColor(type.score);
            
            return (
              <div 
                key={type.name} 
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 hover:scale-105 hover:rotate-1 animate-in slide-in-from-bottom-4 duration-500"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold text-gray-900">
                    {type.name === "Gender Bias" ? "👥 Gender Bias" : 
                     type.name === "Racial Bias" ? "🌍 Racial Bias" :
                     type.name === "Political Bias" ? "🏛️ Political Bias" : 
                     "🎭 Cultural Bias"}
                  </h4>
                  <span className="text-3xl font-bold bg-gradient-to-r from-gray-700 to-gray-900 bg-clip-text text-transparent">
                    {type.score.toFixed(1)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-4 overflow-hidden">
                  <div 
                    className={`${progressColor} h-3 rounded-full transition-all duration-1000 ease-out animate-pulse`}
                    style={{ 
                      width: `${(type.score / 10) * 100}%`,
                      animationDelay: `${index * 200}ms`
                    }}
                  />
                </div>
                <Badge className={`${bias.bgColor} ${bias.color} font-semibold px-3 py-1 animate-bounce`}>
                  {bias.level}
                </Badge>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
