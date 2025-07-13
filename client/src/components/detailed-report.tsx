import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp, Download, FileText, AlertTriangle, CheckCircle, Lightbulb } from "lucide-react";
import { BiasAnalysis } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

interface DetailedReportProps {
  analysis: BiasAnalysis;
}

export default function DetailedReport({ analysis }: DetailedReportProps) {
  const [openSections, setOpenSections] = useState<Set<string>>(new Set());
  const { toast } = useToast();

  const toggleSection = (sectionType: string) => {
    const newOpenSections = new Set(openSections);
    if (newOpenSections.has(sectionType)) {
      newOpenSections.delete(sectionType);
    } else {
      newOpenSections.add(sectionType);
    }
    setOpenSections(newOpenSections);
  };

  const handleExport = () => {
    toast({
      title: "Export Started",
      description: "Your bias report is being prepared for download.",
    });
  };

  const report = analysis.detailedReport;
  const sections = report.sections || [];

  return (
    <Card className="bg-gradient-to-br from-white via-blue-50/30 to-purple-50/30 rounded-3xl shadow-xl border border-gray-200/50 mb-8 backdrop-blur-sm animate-in slide-in-from-bottom-6 duration-700 hover:shadow-2xl hover:scale-[1.01] transition-all">
      <CardContent className="p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-3 hover:scale-110 transition-transform duration-300">
              <FileText className="text-white h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                📋 Detailed Bias Report
              </h2>
              <p className="text-sm text-gray-600">Comprehensive analysis breakdown</p>
            </div>
          </div>
          <Button 
            onClick={handleExport}
            className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-semibold py-2 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 hover:rotate-1 active:scale-95 shadow-lg"
          >
            <Download className="h-4 w-4 mr-2" />
            📥 Export Report
          </Button>
        </div>

        <div className="space-y-6">
          {sections.map((section: any, index: number) => {
            const isOpen = openSections.has(section.type);
            const riskLevel = section.riskLevel || "medium";
            const badgeColor = riskLevel === "high" ? "bg-red-100 text-red-800 border-red-200" : 
                              riskLevel === "medium" ? "bg-yellow-100 text-yellow-800 border-yellow-200" : 
                              "bg-green-100 text-green-800 border-green-200";
            const iconColor = riskLevel === "high" ? "text-red-600" : 
                             riskLevel === "medium" ? "text-yellow-600" : 
                             "text-green-600";

            return (
              <Collapsible 
                key={`${section.type}-${index}`} 
                open={isOpen}
                className="animate-in slide-in-from-left-4 duration-500"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <CollapsibleTrigger
                  className="w-full"
                  onClick={() => toggleSection(section.type)}
                >
                  <div className="w-full bg-white/80 backdrop-blur-sm px-6 py-5 rounded-t-2xl border border-gray-200/50 flex items-center justify-between hover:bg-white/90 transition-all duration-300 hover:shadow-md hover:scale-[1.01] hover:rotate-1">
                    <div className="flex items-center space-x-4">
                      <div className={`p-2 rounded-lg ${badgeColor.replace('text-', 'bg-').replace('bg-', 'bg-').replace('-800', '-100')}`}>
                        <AlertTriangle className={`h-4 w-4 ${iconColor}`} />
                      </div>
                      <div className="text-left">
                        <div className="flex items-center space-x-3">
                          <Badge className={`${badgeColor} font-semibold px-3 py-1`}>
                            {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk
                          </Badge>
                          <h3 className="font-bold text-gray-900 capitalize text-lg">
                            {section.type === "gender" ? "👥 Gender Bias" : 
                             section.type === "racial" ? "🌍 Racial Bias" :
                             section.type === "political" ? "🏛️ Political Bias" : 
                             section.type === "cultural" ? "🎭 Cultural Bias" : 
                             `${section.type} Bias`} Detection
                          </h3>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">
                          {section.issues?.length || 0} issues found
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${riskLevel === "high" ? "bg-red-500" : riskLevel === "medium" ? "bg-yellow-500" : "bg-green-500"}`}></div>
                      {isOpen ? (
                        <ChevronUp className="h-5 w-5 text-gray-500" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-gray-500" />
                      )}
                    </div>
                  </div>
                </CollapsibleTrigger>
                
                <CollapsibleContent>
                  <div className="px-6 py-6 border-l border-r border-b border-gray-200/50 rounded-b-2xl bg-white/60 backdrop-blur-sm animate-in slide-in-from-bottom-4 duration-300">
                    {section.issues && section.issues.length > 0 && (
                      <div className="mb-6">
                        <div className="flex items-center space-x-2 mb-4">
                          <AlertTriangle className="h-5 w-5 text-red-500" />
                          <h4 className="font-bold text-gray-900">🔍 Identified Issues</h4>
                        </div>
                        <ul className="space-y-4">
                          {section.issues.map((issue: any, issueIndex: number) => (
                            <li key={issueIndex} className="flex items-start space-x-3 bg-red-50/50 rounded-lg p-4 hover:bg-red-50/80 transition-all duration-300 hover:scale-[1.01] hover:shadow-md animate-in slide-in-from-left-4"
                                style={{ animationDelay: `${issueIndex * 50}ms` }}>
                              <div className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></div>
                              <div className="flex-1">
                                <p className="text-gray-900 font-medium mb-1">
                                  {issue.description || issue.text || "Issue identified"}
                                </p>
                                {issue.location && issue.location !== "N/A" && (
                                  <p className="text-sm text-gray-600 mb-2">
                                    <span className="font-medium">Location:</span> {issue.location}
                                  </p>
                                )}
                                {issue.severity && (
                                  <Badge variant={issue.severity === "high" ? "destructive" : issue.severity === "medium" ? "secondary" : "outline"} className="text-xs">
                                    {issue.severity} severity
                                  </Badge>
                                )}
                              </div>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {section.recommendations && section.recommendations.length > 0 && section.recommendations.filter((rec: string) => rec !== "N/A").length > 0 && (
                      <div className="mb-6">
                        <div className="flex items-center space-x-2 mb-4">
                          <Lightbulb className="h-5 w-5 text-blue-500" />
                          <h4 className="font-bold text-gray-900">Recommendations</h4>
                        </div>
                        <ul className="space-y-2 bg-blue-50/50 rounded-lg p-4">
                          {section.recommendations.filter((rec: string) => rec !== "N/A").map((rec: string, recIndex: number) => (
                            <li key={recIndex} className="flex items-start space-x-2">
                              <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                              <span className="text-gray-700">{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {section.suggestedRevision && section.suggestedRevision !== "N/A" && (
                      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200/50">
                        <div className="flex items-center space-x-2 mb-3">
                          <FileText className="h-5 w-5 text-green-600" />
                          <h4 className="font-bold text-gray-900">Suggested Revision</h4>
                        </div>
                        <p className="text-gray-800 italic bg-white/60 rounded-lg p-4 border-l-4 border-green-500">
                          "{section.suggestedRevision}"
                        </p>
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            );
          })}
        </div>

        {report.summary && (
          <div className="mt-8 bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 rounded-2xl p-8 border border-blue-200/50 animate-in slide-in-from-bottom-4 duration-700">
            <div className="flex items-center space-x-3 mb-6">
              <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-3">
                <CheckCircle className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900">Summary & Action Items</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200/50">
                <div className="flex items-center space-x-2 mb-4">
                  <AlertTriangle className="h-5 w-5 text-orange-500" />
                  <h4 className="font-bold text-gray-900">Priority Actions</h4>
                </div>
                <ul className="space-y-2">
                  {report.summary.priorityActions?.map((action: string, index: number) => (
                    <li key={index} className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-gray-700">{action}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200/50">
                <div className="flex items-center space-x-2 mb-4">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <h4 className="font-bold text-gray-900">Strengths</h4>
                </div>
                <ul className="space-y-2">
                  {report.summary.strengths?.map((strength: string, index: number) => (
                    <li key={index} className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-gray-700">{strength}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200/50">
                <div className="flex items-center space-x-2 mb-4">
                  <Lightbulb className="h-5 w-5 text-blue-500" />
                  <h4 className="font-bold text-gray-900">Expected Impact</h4>
                </div>
                <p className="text-gray-700 leading-relaxed">{report.summary.impact}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
