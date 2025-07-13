import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { BiasAnalysis } from "@shared/schema";
import { FileText, Newspaper, MessageSquare, Eye, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow } from "date-fns";

export default function AnalysisHistory() {
  const { toast } = useToast();
  
  const { data: analyses, isLoading } = useQuery<BiasAnalysis[]>({
    queryKey: ["/api/analyses"],
    queryFn: async () => {
      const response = await fetch("/api/analyses");
      if (!response.ok) {
        throw new Error("Failed to fetch analysis history");
      }
      return response.json();
    },
  });

  const getIcon = (content: string) => {
    const lowerContent = content.toLowerCase();
    if (lowerContent.includes("news") || lowerContent.includes("article")) {
      return <Newspaper className="text-primary h-5 w-5" />;
    }
    if (lowerContent.includes("social") || lowerContent.includes("post")) {
      return <MessageSquare className="text-primary h-5 w-5" />;
    }
    return <FileText className="text-primary h-5 w-5" />;
  };

  const getScoreColor = (score: number) => {
    if (score <= 3) return "text-green-600";
    if (score <= 6) return "text-yellow-600";
    return "text-red-600";
  };

  const handleViewReport = (id: number) => {
    toast({
      title: "View Report",
      description: `Viewing full report for analysis #${id}`,
    });
  };

  const handleDownloadReport = (id: number) => {
    toast({
      title: "Download Started",
      description: `Downloading report for analysis #${id}`,
    });
  };

  if (isLoading) {
    return (
      <Card className="bg-white rounded-2xl shadow-sm border border-gray-200">
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-6 bg-gray-200 rounded w-1/4"></div>
            <div className="space-y-3">
              <div className="h-16 bg-gray-200 rounded"></div>
              <div className="h-16 bg-gray-200 rounded"></div>
              <div className="h-16 bg-gray-200 rounded"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-white via-blue-50/30 to-purple-50/30 rounded-3xl shadow-xl border border-gray-200/50 backdrop-blur-sm">
      <CardContent className="p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-3">
              <FileText className="text-white h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                Analysis History
              </h2>
              <p className="text-sm text-gray-600">Recent bias analysis reports</p>
            </div>
          </div>
          <Button 
            variant="ghost" 
            className="text-primary hover:text-blue-700 bg-white/60 backdrop-blur-sm hover:bg-white/80 rounded-xl font-semibold"
          >
            View All
          </Button>
        </div>

        <div className="space-y-4">
          {analyses && analyses.length > 0 ? (
            analyses.map((analysis, index) => (
              <div
                key={analysis.id}
                className="flex items-center justify-between p-6 bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 animate-in slide-in-from-left-4 duration-500"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl flex items-center justify-center border border-blue-200/50">
                      {getIcon(analysis.content)}
                    </div>
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-900">
                      {analysis.analysisType.charAt(0).toUpperCase() + analysis.analysisType.slice(1)} Analysis
                    </h3>
                    <div className="flex items-center space-x-3 mt-1">
                      <span className="text-sm text-gray-500">
                        {formatDistanceToNow(new Date(analysis.createdAt), { addSuffix: true })}
                      </span>
                      <span className="text-sm text-gray-500">•</span>
                      <span className="text-sm text-gray-500">{analysis.wordCount} words</span>
                      <span className="text-sm text-gray-500">•</span>
                      <span className={`text-sm font-semibold ${getScoreColor(analysis.overallScore)}`}>
                        Score: {analysis.overallScore.toFixed(1)}/10
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleViewReport(analysis.id)}
                    className="hover:bg-blue-50 rounded-xl"
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDownloadReport(analysis.id)}
                    className="hover:bg-green-50 rounded-xl"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-12 text-gray-500">
              <div className="bg-gradient-to-br from-gray-100 to-gray-200 rounded-2xl p-8 max-w-md mx-auto">
                <FileText className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                <p className="text-lg font-semibold text-gray-600 mb-2">No analysis history yet</p>
                <p className="text-sm text-gray-500">Start by analyzing your first piece of content</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
