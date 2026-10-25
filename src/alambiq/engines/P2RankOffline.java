import cz.siret.prank.program.Main;
import org.biojava.nbio.structure.chem.ChemCompGroupFactory;
import org.biojava.nbio.structure.chem.ReducedChemCompProvider;

/** Launch P2Rank with BioJava's bundled reduced chemistry and no CCD downloads. */
public final class P2RankOffline {
    private P2RankOffline() {}

    public static void main(String[] args) {
        ChemCompGroupFactory.setChemCompProvider(new ReducedChemCompProvider());
        Main.main(args);
    }
}
