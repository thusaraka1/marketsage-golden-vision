import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import PredictionSection from "@/components/PredictionSection";
import InfoSection from "@/components/InfoSection";
import HeatmapSection from "@/components/HeatmapSection";
import ReportsSection from "@/components/ReportsSection";
import NewsFeed from "@/components/NewsFeed";
import ReviewsSection from "@/components/ReviewsSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-void">
      <Header />
      <main>
        <HeroSection />
        <PredictionSection />
        <InfoSection />
        <HeatmapSection />
        <ReportsSection />
        <NewsFeed />
        <ReviewsSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;


