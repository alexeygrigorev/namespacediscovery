package com.formulasearchengine.waugma;

import net.sourceforge.jwbf.core.contentRep.Article;
import net.sourceforge.jwbf.mediawiki.bots.MediaWikiBot;

/**
 * Created by Moritz on 27.09.2015.
 */
public class Main {
	public static void main(String[] args) {
		MediaWikiBot wikiBot = new MediaWikiBot("http://drmf.wmflabs.org/w/");
		Article article = wikiBot.getArticle("Main_Page");
		System.out.println(article.getText());
		// HITCHHIKER'S GUIDE TO THE GALAXY FANS
		applyChangesTo(article);
		//wikiBot.login("user", "***");
		//article.save();
	}

	static void applyChangesTo(Article article) {

		// edits the article...
	}

}
